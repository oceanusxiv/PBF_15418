#include "ParticleSystemSerial.h"
#include "config.h"
#include "clipp.h"
#include "glRenderer.h"
#include "glWindow.h"
#ifdef USE_CUDA
#include "ParticleSystemCUDA.h"
#endif

void printCudaInfo();

int main(int argc, char *argv[]) {

  uint32_t num = 2000, width = 1920, height = 1080, bounds_x = 30, bounds_y = 30, bounds_z = 30;
  std::string config = "dam";
  bool help = false;

  auto cli = (
      clipp::option("-h", "--help").set(help),
      clipp::option("-n", "--number").doc("number of particles") & clipp::value("num", num),
      clipp::option("-r", "--resolution").doc("window size") & clipp::value("x", width) & clipp::value("y", height),
      clipp::option("-b", "--bounds").doc("simulation boundary")
      & clipp::value("x", bounds_x) & clipp::value("y", bounds_y) & clipp::value("z", bounds_z)
  );

  if (!clipp::parse(argc, argv, cli)) {
    std::cout << clipp::usage_lines(cli, "PBF_15418") << std::endl;
    return 1;
  }

  if (help) {
    std::cout <<  clipp::make_man_page(cli, "PBF_15418");
    return 0;
  }

#ifdef USE_CUDA
  ParticleSystemCUDA sim(num, glm::vec3(bounds_x, bounds_y, bounds_z), config);
#else
  ParticleSystemSerial sim(numParticles, glm::vec3(bounds_x, bounds_y, bounds_z), config);
#endif

  glWindow simWindow(width, height);
  simWindow.init();
  glRenderer simRenderer(width, height, simWindow.getCamera(), sim);
  simRenderer.init();

  double lastTime = glfwGetTime();
  int nbFrames = 0;

  while (!glfwWindowShouldClose(simWindow.getWindow())) {
    double currentTime = glfwGetTime();
    nbFrames++;
    if (currentTime - lastTime >= 1.0) {  // If last prinf() was more than 1 sec ago
      std::cout << (currentTime - lastTime) * 1000 / double(nbFrames) << " ms/frame" << std::endl;
      nbFrames = 0;
      lastTime += 1.0;
    }

    simWindow.updateTime();
    glfwPollEvents();
    simWindow.doMovement();
    sim.step();
    simRenderer.onDraw();

    glfwSwapBuffers(simWindow.getWindow());
  }
  glfwTerminate();
  return 0;
}
