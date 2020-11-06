#include <iostream> 
#include <fstream>
#include <random>
#include <algorithm>
#include <math.h>
#include <ctime>
#include <string>
#include <vector>
#include <chrono>
#include "saruprng.hpp"
#include "muller_brown.h"

int main(int argc, char* argv[]) {
    MullerBrown system;
    system.GetParams("param");
    system.Simulate(system.cycles);
    system.DumpPhi();
    system.DumpStates();
}
