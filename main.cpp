#include "glWindow.h"
#include "glRenderer.h"

void printCudaInfo();

int main(int argc, char *argv[]) {

    int numParticles = 2000;
    int width = 640;
    int height = 480;
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

    ParticleSystemCUDA sim(numParticles, glm::vec3(30, 30, 30));

    glWindow simWindow(width, height);
    simWindow.init();

    glRenderer simRenderer(width, height, simWindow.getCamera(), sim, srcPath);
    simRenderer.init();

    double lastTime = glfwGetTime();
    int nbFrames = 0;

    while(!glfwWindowShouldClose(simWindow.getWindow()))
    {
        double currentTime = glfwGetTime();
        nbFrames++;
        if ( currentTime - lastTime >= 1.0 ){ // If last prinf() was more than 1 sec ago
            std::cout << 1000.0/double(nbFrames) << " ms/frame" << std::endl;
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
