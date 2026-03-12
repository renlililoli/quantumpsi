#include <chrono>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <libint2.hpp>

#include "psi4/psi4-dec.h"
#include "psi4/libciomr/libciomr.h"
#include "psi4/libfunctional/superfunctional.h"
#include "psi4/libmints/basisset.h"
#include "psi4/libmints/matrix.h"
#include "psi4/libmints/element_to_Z.h"
#include "psi4/libmints/gshell.h"
#include "psi4/libmints/molecule.h"
#include "psi4/libmints/wavefunction.h"
#include "psi4/liboptions/liboptions.h"
#include "psi4/libpsi4util/PsiOutStream.h"
#include "psi4/libpsi4util/process.h"
#include "psi4/libfock/jk.h"
#include "psi4/libpsio/psio.h"
#include "psi4/libpsio/psio.hpp"
#include "psi4/libqt/qt.h"
#include "psi4/libscf_solver/rhf.h"

namespace psi {
int read_options(const std::string& name, Options& options, bool suppress_printing = false);

// Standalone executable definitions for the legacy globals that are otherwise
// provided by core.cc in the Python module build.
char* psi_file_prefix = nullptr;
std::string outfile_name;
std::string restart_id;
std::shared_ptr<PsiOutStream> outfile;
}

namespace {

using Clock = std::chrono::steady_clock;
using ShellMap = std::map<std::string, std::map<std::string, std::vector<psi::ShellInfo>>>;

struct Config {
    std::string case_file = "benchmark/scf/cases/benzene_dimer.xyz";
    std::string output_file = "stdout";
    std::string csv_file;
    std::string scratch_dir;
    int threads = 1;
    int repeat = 1;
    long memory_mib = 1024;
    std::string scf_type = "DIRECT";
};

std::string uppercase(std::string text) {
    for (char& c : text) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    return text;
}

inline bool file_exists(const std::string& path) {
    std::ifstream f(path);
    return f.good();
}

/** Resolve case file path. Tries path as-is, then BENCHMARK_CASES_DIR (from CMake), then PSI4_BENCHMARK_ROOT env. */
std::string resolve_case_file(const std::string& path) {
    if (file_exists(path)) return path;
    /* Fallback only when using the default case file name */
    const bool is_default = (path.find("benzene_dimer.xyz") != std::string::npos);
    if (!is_default)
        throw std::runtime_error("Case file not found: " + path);
#ifdef BENCHMARK_CASES_DIR
    {
        std::string alt = std::string(BENCHMARK_CASES_DIR) + "/benzene_dimer.xyz";
        if (file_exists(alt)) return alt;
    }
#endif
    const char* root = std::getenv("PSI4_BENCHMARK_ROOT");
    if (root) {
        std::string alt = std::string(root) + "/benchmark/scf/cases/benzene_dimer.xyz";
        if (file_exists(alt)) return alt;
    }
    throw std::runtime_error(
        "Case file not found: " + path +
        "\n  Run from project root, or set PSI4_BENCHMARK_ROOT, or pass --case-file <path>");
}

void print_usage(const char* argv0) {
    std::cerr
        << "Usage: " << argv0 << " [options]\n"
        << "  --case-file <path>     XYZ geometry file. Default: benchmark/scf/cases/benzene_dimer.xyz\n"
        << "  --output-file <path>   Psi4 output file. Use stdout for console output. Default: stdout\n"
        << "  --csv-file <path>      Optional CSV append target for summary rows\n"
        << "  --scratch-dir <path>   Optional PSIO scratch directory\n"
        << "  --threads <n>          OpenMP / MKL thread count. Default: 1\n"
        << "  --repeat <n>           Number of identical runs in one process. Default: 1\n"
        << "  --memory-mib <n>       Psi4 memory limit in MiB. Default: 1024\n"
        << "  --scf-type <type>      DIRECT or PK. Default: DIRECT\n";
}

Config parse_args(int argc, char** argv) {
    Config config;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];

        auto next_value = [&](const char* flag) -> std::string {
            if (i + 1 >= argc) throw std::runtime_error(std::string("Missing value after ") + flag);
            return argv[++i];
        };

        if (arg == "--case-file") {
            config.case_file = next_value("--case-file");
        } else if (arg == "--output-file") {
            config.output_file = next_value("--output-file");
        } else if (arg == "--csv-file") {
            config.csv_file = next_value("--csv-file");
        } else if (arg == "--scratch-dir") {
            config.scratch_dir = next_value("--scratch-dir");
        } else if (arg == "--threads") {
            config.threads = std::stoi(next_value("--threads"));
        } else if (arg == "--repeat") {
            config.repeat = std::stoi(next_value("--repeat"));
        } else if (arg == "--memory-mib") {
            config.memory_mib = std::stol(next_value("--memory-mib"));
        } else if (arg == "--scf-type") {
            config.scf_type = uppercase(next_value("--scf-type"));
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        } else {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }

    if (config.threads < 1) throw std::runtime_error("--threads must be >= 1");
    if (config.repeat < 1) throw std::runtime_error("--repeat must be >= 1");
    if (config.memory_mib < 1) throw std::runtime_error("--memory-mib must be >= 1");
    if (config.scf_type != "DIRECT" && config.scf_type != "PK") {
        throw std::runtime_error("--scf-type must be DIRECT or PK");
    }

    return config;
}

#define DBG(...) do { std::cerr << "[DBG] "; std::cerr << __VA_ARGS__ << std::endl; } while (0)
#define DBG_FLUSH(...) do { std::cerr << "[DBG] "; std::cerr << __VA_ARGS__ << std::flush; } while (0)

void initialize_runtime(const Config& config) {
    DBG("initialize_runtime: Wavefunction::initialize_singletons");
    psi::Wavefunction::initialize_singletons();
    DBG("initialize_runtime: BasisSet::initialize_singletons");
    psi::BasisSet::initialize_singletons();

    psi::outfile_name = config.output_file;
    if (config.output_file == "stdout") {
        psi::outfile = std::make_shared<psi::PsiOutStream>();
    } else {
        psi::outfile = std::make_shared<psi::PsiOutStream>(config.output_file, std::ostream::trunc);
    }

    const std::string prefix = "psi";
    psi::psi_file_prefix = strdup(prefix.c_str());

    DBG("initialize_runtime: timer_init");
    psi::timer_init();
    DBG("initialize_runtime: psio_init");
    psi::psio_init();
    DBG("initialize_runtime: Process::environment.initialize");
    psi::Process::environment.initialize();
    psi::Process::environment.set_memory(static_cast<size_t>(config.memory_mib) * 1024ULL * 1024ULL);
    psi::Process::environment.set_n_threads(config.threads);

    if (!config.scratch_dir.empty()) {
        psi::PSIOManager::shared_object()->set_default_path(config.scratch_dir);
    }

    DBG("initialize_runtime: read global options");
    auto& options = psi::Process::environment.options;
    options.set_read_globals(true);
    psi::read_options("", options, true);
    options.set_read_globals(false);
    DBG("initialize_runtime: read SCF options");
    options.set_current_module("SCF");
    psi::read_options("SCF", options, true);
    DBG("initialize_runtime: validate_options");
    options.validate_options();
    DBG("initialize_runtime: done");
}

void configure_libint() {
    DBG("configure_libint: start");
    // Match the Psi4 build's solid-harmonic ordering when that compile-time
    // setting is available. Fall back to the common standard ordering otherwise.
#if defined(psi4_SHGSHELL_ORDERING) && (psi4_SHGSHELL_ORDERING == LIBINT_SHGSHELL_ORDERING_GAUSSIAN)
    libint2::set_solid_harmonics_ordering(libint2::SHGShellOrdering_Gaussian);
#else
    libint2::set_solid_harmonics_ordering(libint2::SHGShellOrdering_Standard);
#endif
    libint2::initialize();
    DBG("configure_libint: done");
}

void finalize_runtime(bool libint_initialized) {
    psi::PSIOManager::shared_object()->psiclean();
    psi::timer_done();
    if (libint_initialized) libint2::finalize();
    psi::outfile.reset();
    if (psi::psi_file_prefix) {
        free(psi::psi_file_prefix);
        psi::psi_file_prefix = nullptr;
    }
}

std::shared_ptr<psi::Molecule> load_xyz_molecule(const std::string& path) {
    DBG("load_xyz_molecule: " << path);
    std::ifstream input(path);
    if (!input) throw std::runtime_error("Failed to open XYZ file: " + path);

    std::size_t natom = 0;
    input >> natom;
    if (!input) throw std::runtime_error("Failed to read atom count from XYZ file: " + path);

    std::string line;
    std::getline(input, line);
    std::getline(input, line);

    psi::Element_to_Z element_to_z;
    auto molecule = std::make_shared<psi::Molecule>();
    molecule->set_name("benchmark_case");
    molecule->set_units(psi::Molecule::Angstrom);
    molecule->set_molecular_charge(0);
    molecule->set_multiplicity(1);
    molecule->set_orientation_fixed(true);
    molecule->set_com_fixed(true);

    for (std::size_t i = 0; i < natom; ++i) {
        std::string symbol;
        double x = 0.0;
        double y = 0.0;
        double z = 0.0;
        input >> symbol >> x >> y >> z;
        if (!input) throw std::runtime_error("Failed to read XYZ atom line from: " + path);

        const std::string upper_symbol = uppercase(symbol);
        const double Z = element_to_z[upper_symbol];
        if (Z <= 0.0) throw std::runtime_error("Unsupported element in XYZ file: " + symbol);

        molecule->add_atom(Z, x, y, z, upper_symbol, 0.0, 0.0, upper_symbol);
    }

    /* Must set fragment pattern before update_geometry(); otherwise reinterpret_coordentries()
     * clears atoms_ and never repopulates it (fragments_ empty => natom() becomes 0). */
    molecule->set_fragment_pattern({{0, static_cast<int>(natom)}}, {psi::Molecule::Real},
                                  {0}, {1});
    DBG("load_xyz_molecule: update_geometry, natom=" << natom);
    molecule->update_geometry();
    DBG("load_xyz_molecule: done");
    return molecule;
}

std::vector<psi::ShellInfo> carbon_631g_d_shells() {
    using psi::Cartesian;
    using psi::ShellInfo;
    using psi::Unnormalized;

    return {
        ShellInfo(0, {0.0018347, 0.0140373, 0.0688426, 0.2321844, 0.4679413, 0.3623120},
                  {3047.5249000, 457.3695100, 103.9486900, 29.2101550, 9.2866630, 3.1639270},
                  Cartesian, Unnormalized),
        ShellInfo(0, {-0.1193324, -0.1608542, 1.1434564},
                  {7.8682724, 1.8812885, 0.5442493},
                  Cartesian, Unnormalized),
        ShellInfo(1, {0.0689991, 0.3164240, 0.7443083},
                  {7.8682724, 1.8812885, 0.5442493},
                  Cartesian, Unnormalized),
        ShellInfo(0, {1.0}, {0.1687144}, Cartesian, Unnormalized),
        ShellInfo(1, {1.0}, {0.1687144}, Cartesian, Unnormalized),
        ShellInfo(2, {1.0}, {0.8000000}, Cartesian, Unnormalized),
    };
}

std::vector<psi::ShellInfo> hydrogen_631g_d_shells() {
    using psi::Cartesian;
    using psi::ShellInfo;
    using psi::Unnormalized;

    return {
        ShellInfo(0, {0.03349460, 0.23472695, 0.81375733},
                  {18.7311370, 2.8253937, 0.6401217},
                  Cartesian, Unnormalized),
        ShellInfo(0, {1.0}, {0.1612778}, Cartesian, Unnormalized),
    };
}

std::shared_ptr<psi::BasisSet> build_embedded_631g_d_basis(const std::shared_ptr<psi::Molecule>& molecule) {
    const std::string basis_name = "EMBEDDED_631G_D";
    ShellMap shell_map;
    ShellMap ecp_shell_map;

    shell_map[basis_name]["C"] = carbon_631g_d_shells();
    shell_map[basis_name]["H"] = hydrogen_631g_d_shells();

    molecule->set_basis_all_atoms(basis_name, "BASIS");
    molecule->set_shell_by_label("C", "C", "BASIS");
    molecule->set_shell_by_label("H", "H", "BASIS");
    molecule->update_geometry();

    auto basis = std::make_shared<psi::BasisSet>("BASIS", molecule, shell_map, ecp_shell_map);
    basis->set_name(basis_name);
    basis->set_key("BASIS");
    basis->set_target(basis_name);
    return basis;
}

std::shared_ptr<psi::SuperFunctional> build_hf_superfunctional() {
    auto functional = psi::SuperFunctional::blank();
    functional->set_name("HF");
    functional->set_description("Minimal HF superfunctional for the standalone SCF benchmark");
    functional->set_x_alpha(1.0);
    functional->set_lock(true);
    return functional;
}

/** Run SCF iterations. Replicates Python scf_compute_energy logic for standalone build. */
double run_scf(std::shared_ptr<psi::scf::RHF> rhf) {
    DBG("run_scf: start");
    auto& options = psi::Process::environment.options;
    auto basis = rhf->basisset();
    options.set_current_module("SCF");
    const double e_conv = options.get_double("E_CONVERGENCE");
    const double d_conv = options.get_double("D_CONVERGENCE");
    const int maxiter = options.get_int("MAXITER");
    const double level_shift = options.get_double("LEVEL_SHIFT");
    const double level_shift_cutoff = options.get_double("LEVEL_SHIFT_CUTOFF");
    DBG("run_scf: e_conv=" << e_conv << " d_conv=" << d_conv << " maxiter=" << maxiter);

    DBG("run_scf: JK::build_JK");
    auto jk = psi::JK::build_JK(basis, nullptr, options, options.get_str("SCF_TYPE"));
    jk->set_print(rhf->get_print());
    jk->set_do_K(true);
    DBG("run_scf: jk->initialize");
    jk->initialize();
    DBG("run_scf: jk->initialize done, set_jk");
    rhf->set_jk(jk);

    DBG("run_scf: form_H");
    rhf->form_H();
    DBG("run_scf: form_Shalf");
    rhf->form_Shalf();
    DBG("run_scf: guess");
    rhf->guess();

    double e_old = 0.0;
    double e_prev_2 = 0.0;  /* energy from 2 iters ago, for oscillation detection */
    DBG("run_scf: SCF iteration loop start");
    for (int iter = 1; iter <= maxiter; ++iter) {
        rhf->set_iteration(iter);
        rhf->save_density_and_energy();
        rhf->form_G();
        rhf->form_F();
        const double e_new = rhf->compute_E();
        rhf->set_energies("Total Energy", e_new);

        const double de = std::abs(e_new - e_old);
        double dnorm = 0.0;
        if (iter > 1) {
            auto gradient = rhf->form_FDSmSDF(rhf->Fa(), rhf->Da());
            dnorm = gradient->absmax();
        }
        const bool oscillating = (iter >= 6) && (std::abs(e_new - e_prev_2) < 1.0e-4);
        const bool use_level_shift = (level_shift > 0.0) &&
            (dnorm > level_shift_cutoff || oscillating);
        e_prev_2 = e_old;
        e_old = e_new;

        DBG("run_scf: iter=" << iter << " E=" << e_new << " de=" << de << " dnorm=" << dnorm
            << (use_level_shift ? " [LEVEL_SHIFT]" : ""));
        if (de < e_conv && dnorm < d_conv) {
            DBG("run_scf: converged at iter=" << iter);
            break;
        }
        if (use_level_shift) {
            rhf->form_C(level_shift);
        } else {
            rhf->form_C();
        }
        rhf->form_D();
    }

    DBG("run_scf: finalize");
    rhf->finalize();
    return rhf->energy();
}

void configure_options(const Config& config) {
    auto& options = psi::Process::environment.options;

    options.set_global_str("BASIS", "EMBEDDED_631G_D");
    options.set_global_str("SCF_TYPE", config.scf_type);
    options.set_global_str("REFERENCE", "RHF");
    options.set_global_str("FREEZE_CORE", "FALSE");
    options.set_global_int("PRINT", 2);
    options.set_global_bool("PUREAM", false);

    options.set_str("SCF", "GUESS", "CORE");
    options.set_double("SCF", "E_CONVERGENCE", 1.0e-8);
    options.set_double("SCF", "D_CONVERGENCE", 1.0e-8);
    options.set_int("SCF", "MAXITER", 10);
    options.set_double("SCF", "INTS_TOLERANCE", 1.0e-12);
    options.set_double("SCF", "LEVEL_SHIFT", 0.1);
    options.set_double("SCF", "LEVEL_SHIFT_CUTOFF", 1.0e-2);
}

struct RunResult {
    double energy = 0.0;
    double elapsed_seconds = 0.0;
    int iterations = 0;
    int nbf = 0;
};

RunResult run_one_scf(const Config& config) {
    DBG("run_one_scf: configure_options");
    configure_options(config);

    DBG("run_one_scf: resolve_case_file");
    const std::string resolved_path = resolve_case_file(config.case_file);
    auto molecule = load_xyz_molecule(resolved_path);
    if (molecule->natom() == 0)
        throw std::runtime_error("Molecule has no atoms. Case file may be empty or malformed: " + resolved_path);
    DBG("run_one_scf: build_embedded_631g_d_basis");
    auto basis = build_embedded_631g_d_basis(molecule);
    auto reference = std::make_shared<psi::Wavefunction>(molecule, basis, psi::Process::environment.options);
    auto functional = build_hf_superfunctional();

    psi::Process::environment.set_molecule(molecule);
    psi::PSIOManager::shared_object()->psiclean();

    psi::outfile->Printf("\n  Standalone SCF benchmark\n");
    psi::outfile->Printf("  Case file: %s\n", resolved_path.c_str());
    psi::outfile->Printf("  Basis:     %s\n", basis->name().c_str());
    psi::outfile->Printf("  SCF_TYPE:  %s\n", config.scf_type.c_str());
    psi::outfile->Printf("  Threads:   %d\n\n", config.threads);

    DBG("run_one_scf: creating RHF");
    auto rhf = std::make_shared<psi::scf::RHF>(reference, functional);

    const auto t0 = Clock::now();
    DBG("run_one_scf: calling run_scf");
    const double energy = run_scf(rhf);
    const auto t1 = Clock::now();

    RunResult result;
    result.energy = energy;
    result.elapsed_seconds = std::chrono::duration<double>(t1 - t0).count();
    result.iterations = rhf->iteration();
    result.nbf = basis->nbf();
    return result;
}

void append_csv(const Config& config, const RunResult& result, int run_index) {
    if (config.csv_file.empty()) return;

    const bool write_header = !std::ifstream(config.csv_file).good();
    std::ofstream csv(config.csv_file, std::ios::app);
    if (!csv) throw std::runtime_error("Failed to open CSV file: " + config.csv_file);

    if (write_header) {
        csv << "run,case_file,basis,scf_type,threads,repeat_index,nbf,iterations,energy_hartree,elapsed_seconds\n";
    }

    csv << "scf"
        << "," << config.case_file
        << "," << "EMBEDDED_631G_D"
        << "," << config.scf_type
        << "," << config.threads
        << "," << run_index
        << "," << result.nbf
        << "," << result.iterations
        << "," << std::setprecision(16) << result.energy
        << "," << std::setprecision(8) << result.elapsed_seconds
        << "\n";
}

}  // namespace

int main(int argc, char** argv) {
    bool runtime_initialized = false;
    bool libint_initialized = false;

    try {
        DBG("main: parse_args");
        const Config config = parse_args(argc, argv);
        DBG("main: case_file=" << config.case_file << " scf_type=" << config.scf_type << " threads=" << config.threads);

        /* Avoid SIGSEGV in mkl_set_num_threads when MKL isn't initialized yet in
         * standalone build. Set env vars so MKL/OMP use the right thread count. */
        {
            char buf[32];
            std::snprintf(buf, sizeof(buf), "%d", config.threads);
            setenv("MKL_NUM_THREADS", buf, 1);
            setenv("OMP_NUM_THREADS", buf, 1);
            setenv("PSI4_SKIP_MKL_SET_NUM_THREADS", "1", 1);
        }

        DBG("main: initialize_runtime");
        initialize_runtime(config);
        runtime_initialized = true;
        DBG("main: configure_libint");
        configure_libint();
        libint_initialized = true;

        DBG("main: entering run loop, repeat=" << config.repeat);
        for (int run = 1; run <= config.repeat; ++run) {
            DBG("main: run " << run << "/" << config.repeat);
            const RunResult result = run_one_scf(config);
            DBG("main: run " << run << " done, energy=" << result.energy);
            append_csv(config, result, run);

            std::cout << "run=" << run
                      << " threads=" << config.threads
                      << " nbf=" << result.nbf
                      << " iterations=" << result.iterations
                      << " energy=" << std::setprecision(16) << result.energy
                      << " elapsed_s=" << std::setprecision(8) << result.elapsed_seconds
                      << std::endl;
        }

        finalize_runtime(libint_initialized);
        return 0;
    } catch (const std::exception& ex) {
        if (runtime_initialized) finalize_runtime(libint_initialized);
        std::cerr << "SCF benchmark failed: " << ex.what() << std::endl;
        return 1;
    }
}
