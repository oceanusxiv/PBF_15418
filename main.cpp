#include "glWindow.h"
#include "glRenderer.h"

void saxpyCuda(int N, float alpha, float* x, float* y, float* result);

int main(int argc, char *argv[]) {
    int N = 20 * 1000;
    const float alpha = 2.0f;
    float* xarray = new float[N];
    float* yarray = new float[N];
    float* resultarray = new float[N];

    // load X, Y, store result
    for (int i=0; i<N; i++) {
        xarray[i] = yarray[i] = i % 10;
        resultarray[i] = 0.f;
    }

    std::cout << "STARTING SAXPY\n";
    saxpyCuda(N, alpha, xarray, yarray, resultarray);
    std::cout << "DONE SAXPY\n";
    for (int i = 0; i < 10; i++) {
      std::cout << resultarray[i] << std::endl;
    }

    delete [] xarray;
    delete [] yarray;
    delete [] resultarray;

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

    particleSystem sim(numParticles, glm::vec3(20, 20, 20));

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
