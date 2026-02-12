#include <iostream>
#include <string>
#include <vector>

// MOAB Includes
#include "moab/Core.hpp"
#include "moab/GeomTopoTool.hpp"
#include "moab/Interface.hpp"

int main(int argc, char *argv[]) {
  // 1. Basic Argument Parsing
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <filename> [options]" << std::endl;
    std::cerr << "Example: " << argv[0] << " geometry.h5m" << std::endl;
    return 1;
  }

  std::string filename = argv[1];

  // 2. Initialize MOAB Core Instance
  // This is the main database interface
  moab::Core *mb = new moab::Core();

  // 3. Load the File
  std::cout << "[INFO] Loading mesh file: " << filename << "..." << std::endl;

  // We pass NULL for options unless you have specific read options
  moab::ErrorCode rval = mb->load_file(filename.c_str());

  if (rval != moab::MB_SUCCESS) {
    std::cerr << "[ERROR] Failed to load file: " << filename << std::endl;
    std::cerr << "[ERROR] MOAB Error Code: " << mb->get_error_string(rval)
              << std::endl;
    delete mb;
    return 1;
  }
  std::cout << "[INFO] File loaded successfully." << std::endl;

  // 4. Initialize GeomTopoTool
  // The constructor takes the MOAB interface pointer.
  // The second argument 'true' tells it to automatically find and classify
  // geometry sets (Volume, Surface, Curve, Vertex) present in the file.
  std::cout << "[INFO] Initializing Geometry Topology Tool..." << std::endl;
  moab::GeomTopoTool *gtt = new moab::GeomTopoTool(mb, true);

  // 5. Run check_model()
  // This executes the logic you provided: checking vertex sets,
  // edge continuity, and surface skinning.
  std::cout << "--------------------------------------------------------"
            << std::endl;
  std::cout << "[INFO] Running check_model()..." << std::endl;

  bool is_valid = gtt->check_model();

  std::cout << "--------------------------------------------------------"
            << std::endl;

  // 6. Report Final Status
  if (is_valid) {
    std::cout << "[SUCCESS] The model topology appears valid." << std::endl;
  } else {
    std::cout << "[FAILURE] The model failed topology checks." << std::endl;
    std::cout << "          (Review the specific error messages above)"
              << std::endl;
  }

  // 7. Cleanup
  delete gtt;
  delete mb;

  // Return 0 on success, 1 on failure
  return is_valid ? 0 : 1;
}