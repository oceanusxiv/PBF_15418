#include "glWindow.h"
#include "glRenderer.h"

void printCudaInfo();

int main(int argc, char *argv[]) {

    int numParticles = 2000;
    int width = 1280;
    int height = 720;
    std::string srcPath = "./";

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-n") || !strcmp(argv[i], "--num")) {
            numParticles = std::stoi(argv[i + 1]);
        }
        if (!strcmp(argv[i], "-r") || !strcmp(argv[i], "--resolution")) {
            width = std::stoi(argv[i + 1]);
            height = std::stoi(argv[i + 2]);
        }
        if (!strcmp(argv[i], "-p") || !strcmp(argv[i], "--path")) {
            srcPath = argv[i + 1];
        }
    }

    ParticleSystemSerial sim(numParticles, glm::vec3(20, 20, 20));

    glWindow simWindow(width, height);
    simWindow.init();

    glRenderer simRenderer(width, height, simWindow.getCamera(), sim, srcPath);
    simRenderer.init();


    while(!glfwWindowShouldClose(simWindow.getWindow()))
    {
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
