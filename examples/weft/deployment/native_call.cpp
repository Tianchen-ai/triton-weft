// Load the 64-element artifact exported by language/views.py --size 64.
#include <weft/runtime.hpp>

#include <iostream>
#include <numeric>
#include <vector>

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "Usage: weft-native-call /path/to/affine-native\n";
    return 2;
  }
  try {
    std::vector<float> x(64), y(64);
    std::iota(x.begin(), x.end(), 0.0f);
    weft::context device;
    weft::kernel kernel(device, argv[1]);
    auto binding = kernel.bind();
    binding
        .set("X", weft::view(x.data(), x.size() * sizeof(float), "dense.f32",
                             {x.size()}, {}, true))
        .set("Y", weft::view(y.data(), y.size() * sizeof(float), "dense.f32",
                             {y.size()}))
        .set("scale", weft::scalar(1.25f))
        .set("bias", weft::scalar(-0.5f));
    auto call = binding.prepare();
    call();
    std::cout << "Y = 1.25 * X - 0.5:";
    for (size_t i = 0; i < 4; ++i)
      std::cout << ' ' << y[i];
    std::cout << '\n';
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
