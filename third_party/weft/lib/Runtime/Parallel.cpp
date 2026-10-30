// Scheduling and barrier code adapted from ggml-cpu.c in llama.cpp
// 620d2225cc436ab0b001e3200e590e3b286ef9ab (see ggml-threadpool-LICENSE).
// Graph evaluation is replaced by a checked synchronous range callback.
#include "weft/runtime.h"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <exception>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>
#include <sched.h>
#include <unistd.h>

namespace {
inline void relaxCPU() {
#if defined(__x86_64__)
  __asm__ volatile("pause" ::: "memory");
#elif defined(__aarch64__)
  __asm__ volatile("yield" ::: "memory");
#elif defined(__riscv)
  __asm__ volatile(".4byte 0x100000F" ::: "memory");
#else
  std::this_thread::yield();
#endif
}

// Adapted from ggml_barrier: the last worker resets the count and advances a
// generation. Sequentially consistent entry/exit publishes callback outputs.
struct alignas(64) Barrier {
  std::atomic<unsigned> arrived{0};
  std::atomic<uint64_t> passed{0};
  void wait(unsigned threads) {
    if (threads == 1) return;
    uint64_t generation = passed.load(std::memory_order_relaxed);
    if (arrived.fetch_add(1, std::memory_order_seq_cst) == threads - 1) {
      arrived.store(0, std::memory_order_relaxed);
      passed.fetch_add(1, std::memory_order_seq_cst);
      return;
    }
    while (passed.load(std::memory_order_acquire) == generation) relaxCPU();
  }
};

thread_local unsigned parallelDepth = 0;
struct Pool {
  int threads;
  cpu_set_t affinity;
  pid_t owner = getpid();
  std::vector<std::thread> workers;
  std::mutex mutex;
  std::condition_variable condition;
  std::atomic<uint64_t> generation{0};
  std::atomic<bool> stop{false};
  Barrier barrier;
  size_t count = 0;
  const void *scope = nullptr;
  weft_worker_check check = nullptr;
  void *arguments = nullptr;
  weft_parallel_body body = nullptr;
  std::atomic<int> failed{0};
  char diagnostic[1024]{};

  cpu_set_t workerAffinity(int index) const {
    // Keep the inherited CPU set as the resource boundary, but distribute the
    // participating threads rather than leaving barrier partners on one core.
    cpu_set_t selected;
    CPU_ZERO(&selected);
    int count = CPU_COUNT(&affinity);
    int position = count ? index % count : 0;
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu)
      if (CPU_ISSET(cpu, &affinity) && position-- == 0) {
        CPU_SET(cpu, &selected);
        return selected;
      }
    return affinity;
  }

  Pool(int size, const cpu_set_t &mask) : threads(size), affinity(mask) {
    try {
      for (int i = 1; i < threads; ++i) workers.emplace_back([this, i] { secondary(i); });
    } catch (...) {
      shutdown();
      throw;
    }
  }
  ~Pool() { shutdown(); }
  void shutdown() {
    { std::lock_guard<std::mutex> lock(mutex); stop.store(true, std::memory_order_release); }
    condition.notify_all();
    for (auto &worker : workers) if (worker.joinable()) worker.join();
    workers.clear();
  }
  void fail(int status, const char *message) {
    int expected = 0;
    if (failed.compare_exchange_strong(expected, status))
      std::snprintf(diagnostic, sizeof(diagnostic), "%s", message);
  }
  void execute(int index) {
    ++parallelDepth;
    char error[1024]{};
    try {
      int status = check(scope, error, sizeof(error));
      if (status) fail(status, error);
    } catch (...) { fail(WEFT_TARGET_ERROR, "parallel worker check threw an exception"); }
    barrier.wait(threads);
    if (!failed.load(std::memory_order_acquire)) {
      // Quotient/remainder partition avoids overflowing count * worker.
      size_t quotient = count / threads, remainder = count % threads;
      size_t begin = quotient * index + std::min<size_t>(index, remainder);
      size_t end = begin + quotient + (size_t(index) < remainder);
      try { if (begin != end) body(arguments, begin, end); }
      catch (...) { fail(WEFT_RESOURCE_ERROR, "parallel body threw an exception"); }
    }
    barrier.wait(threads);
    --parallelDepth;
  }
  void secondary(int index) {
    cpu_set_t assigned = workerAffinity(index);
    // A failed optional placement leaves the inherited mask in force.
    (void)sched_setaffinity(0, sizeof(assigned), &assigned);
    uint64_t last = 0;
    for (;;) {
      // ggml_graph_compute_poll_for_work / check_for_work: first poll for a
      // new generation, then sleep under the kickoff mutex without lost wakes.
      for (unsigned round = 0; round < 16384 && !stop.load(std::memory_order_relaxed) &&
           generation.load(std::memory_order_acquire) == last; ++round) relaxCPU();
      if (generation.load(std::memory_order_acquire) == last && !stop.load()) {
        std::unique_lock<std::mutex> lock(mutex);
        condition.wait(lock, [&] { return stop.load() || generation.load() != last; });
      }
      if (stop.load(std::memory_order_acquire)) return;
      last = generation.load(std::memory_order_acquire);
      execute(index);
    }
  }
  int run(size_t size, const void *context, weft_worker_check validate,
          void *args, weft_parallel_body invoke, char *message, size_t capacity) {
    count = size; scope = context; check = validate; arguments = args; body = invoke;
    diagnostic[0] = 0; failed.store(0, std::memory_order_relaxed);
    // ggml_graph_compute_kickoff publishes the job with the same mutex used
    // by sleeping workers. The caller participates as worker zero.
    { std::lock_guard<std::mutex> lock(mutex); generation.fetch_add(1, std::memory_order_release); }
    condition.notify_all();
    cpu_set_t assigned = workerAffinity(0);
    bool placed = sched_setaffinity(0, sizeof(assigned), &assigned) == 0;
    execute(0);
    if (placed) (void)sched_setaffinity(0, sizeof(affinity), &affinity);
    int status = failed.load(std::memory_order_acquire);
    if (status && message && capacity) std::snprintf(message, capacity, "%s", diagnostic);
    return status;
  }
};

struct LocalPools {
  pid_t owner = getpid();
  std::vector<Pool *> values;
  void release() {
    // After fork there are no inherited workers to join, and inherited mutexes
    // may belong to vanished threads. Abandon those child-side objects.
    if (owner == getpid()) for (auto *pool : values) delete pool;
    values.clear(); owner = getpid();
  }
  ~LocalPools() { release(); }
  Pool &get(int threads, const cpu_set_t &affinity) {
    if (owner != getpid()) release();
    for (auto *pool : values)
      if (pool->threads == threads && CPU_EQUAL(&pool->affinity, &affinity)) return *pool;
    auto *pool = new Pool(threads, affinity);
    try { values.push_back(pool); } catch (...) { delete pool; throw; }
    return *pool;
  }
};
thread_local LocalPools pools;
} // namespace

extern "C" int weft_parallel_for(size_t count, int threads, const void *scope,
    weft_worker_check check, void *arguments, weft_parallel_body body, char *message, size_t capacity) {
  if (!count) return WEFT_OK;
  auto invalid = [&](const char *text) {
    if (message && capacity) std::snprintf(message, capacity, "%s", text);
    return int(WEFT_INVALID_ARGUMENT);
  };
  if (threads <= 0 || !check || !body) return invalid("parallel execution needs positive threads and callbacks");
  threads = int(std::min<size_t>(threads, count));
  try {
    if (threads == 1 || parallelDepth) {
      int status = check(scope, message, capacity);
      if (!status) body(arguments, 0, count);
      return status;
    }
    cpu_set_t affinity;
    if (sched_getaffinity(0, sizeof(affinity), &affinity))
      return invalid("cannot query parallel execution affinity");
    return pools.get(threads, affinity).run(count, scope, check, arguments, body, message, capacity);
  } catch (const std::exception &error) {
    if (message && capacity) std::snprintf(message, capacity, "%s", error.what());
    return WEFT_RESOURCE_ERROR;
  } catch (...) { return WEFT_RESOURCE_ERROR; }
}

extern "C" void weft_parallel_release() {
  if (!parallelDepth) pools.release();
}
