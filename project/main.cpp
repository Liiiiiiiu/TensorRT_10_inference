#include "build.h"

int main(int argc, char **argv)
{
    if (argc < 4)
    {
        std::cout << "Please design model arch, config file and folder name!" << std::endl;
        return -1;
    }
    std::string folder_name = argv[3];
    auto model = build_model(argv);
    if (model == nullptr)
        return -1;
    std::cout << "2222" << std::endl;
    model->LoadEngine();
    model->InferenceFolder(folder_name);
    return 0;
}
