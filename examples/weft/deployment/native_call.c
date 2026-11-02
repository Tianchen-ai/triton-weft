/* Load the same 64-element View artifact as native_call.cpp. */
#include <weft/runtime.h>
#include <stdio.h>

int main(int argc, char **argv) {
  if (argc != 2) {
    fprintf(stderr, "Usage: weft-native-call-c /path/to/affine-native\n");
    return 2;
  }
  float x[64], y[64] = {0};
  const size_t shape[] = {64};
  for (size_t i = 0; i < 64; ++i)
    x[i] = (float)i;
  weft_value values[] = {
      {.kind = WEFT_VIEW,
       .view = {x, sizeof(x), "dense.f32", 1, shape, NULL, true}},
      {.kind = WEFT_VIEW,
       .view = {y, sizeof(y), "dense.f32", 1, shape, NULL, false}},
      {.kind = WEFT_SCALAR, .dtype = WEFT_F32, .scalar = {.f32 = 1.25f}},
      {.kind = WEFT_SCALAR, .dtype = WEFT_F32, .scalar = {.f32 = -0.5f}}};
  const char *names[] = {"X", "Y", "scale", "bias"};
  weft_error error = {0};
  weft_context *context = NULL;
  weft_kernel *kernel = NULL;
  weft_bindings *bindings = NULL;
  weft_call *call = NULL;
  int result = 1;
#define CHECK(expression)                                                       \
  do {                                                                        \
    if ((expression) != WEFT_OK) {                                             \
      fprintf(stderr, "%s\n", error.message);                                 \
      goto cleanup;                                                           \
    }                                                                         \
  } while (0)
  CHECK(weft_context_create(&context, &error));
  CHECK(weft_kernel_open(context, argv[1], &kernel, &error));
  CHECK(weft_bindings_create(kernel, &bindings, &error));
  for (size_t i = 0; i < 4; ++i)
    CHECK(weft_bindings_set(bindings, names[i], &values[i], &error));
  CHECK(weft_bindings_prepare(bindings, &call, &error));
  CHECK(weft_call_invoke(call, &error));
  for (size_t i = 0; i < 64; ++i) {
    if (y[i] != 1.25f * x[i] - 0.5f) {
      fprintf(stderr, "complete output check failed at %zu\n", i);
      goto cleanup;
    }
  }
  puts("C SDK: all 64 outputs match");
  result = 0;
cleanup:
  if (call) weft_call_release(call);
  if (bindings) weft_bindings_release(bindings);
  if (kernel) weft_kernel_release(kernel);
  if (context) weft_context_release(context);
  return result;
#undef CHECK
}
