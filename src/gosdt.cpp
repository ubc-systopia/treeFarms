#include "gosdt.hpp"
#include "configuration.hpp"

#define _DEBUG true
#define THROTTLE false

float GOSDT::time = 0.0;
unsigned int GOSDT::size = 0;
unsigned int GOSDT::iterations = 0;
unsigned int GOSDT::status = 0;

// https://stackoverflow.com/a/63391159
void tic(const string &task_description, int mode = 0) {
    static std::chrono::time_point<std::chrono::high_resolution_clock> extraction_start;
    static string stored_task_description;
    if (Configuration::verbose) {
        if (mode == 0) {
            extraction_start = std::chrono::high_resolution_clock::now(); // Start measuring training time
            stored_task_description = task_description;
        } else {
            auto extraction_stop = std::chrono::high_resolution_clock::now(); // Stop measuring training time
            float time = std::chrono::duration_cast<std::chrono::milliseconds>(extraction_stop - extraction_start).count() / 1000.0;
            std::cout << stored_task_description << time << " seconds" << std::endl;
        }
    }
}

void toc() {
    tic("", 1);
}

GOSDT::GOSDT(void) {}

GOSDT::~GOSDT(void) {
    return;
}

void GOSDT::configure(std::istream & config_source) { Configuration::configure(config_source); }


void GOSDT::fit(std::istream & data_source) {
    results_t results = results_t();
    fit(data_source, results);
}

void GOSDT::fit(std::istream & data_source, std::string & result) {
    results_t results = results_t();
    fit(data_source, results);
    ModelSet::serialize(results, result, 0);
}


void GOSDT::fit(std::istream & data_source, results_t & results) {
    if(Configuration::verbose) { std::cout << "Using configuration: " << Configuration::to_string(2) << std::endl; }

    if(Configuration::verbose) { std::cout << "Initializing Optimization Framework" << std::endl; }
    Optimizer optimizer;
    optimizer.load(data_source);

    // Dump dataset metadata if requested and terminate early 
    if (Configuration::datatset_encoding != "") {
        json output = json::array();
        for (unsigned int binary_feature_index=0; binary_feature_index<State::dataset.encoder.binary_features(); binary_feature_index++) {
            json node = json::object();
            unsigned int feature_index;
            std::string feature_name, feature_type, relation, reference;
            State::dataset.encoder.decode(binary_feature_index, & feature_index);
            State::dataset.encoder.encoding(binary_feature_index, feature_type, relation, reference);
            State::dataset.encoder.header(feature_index, feature_name);

            node["feature"] = feature_index;
            node["name"] = feature_name;
            node["relation"] = relation;
            if (Encoder::test_integral(reference)) {
                node["type"] = "integral";
                node["reference"] = atoi(reference.c_str());
            } else if (Encoder::test_rational(reference)) {
                node["type"] = "rational";
                node["reference"] = atof(reference.c_str());
            } else {
                node["type"] = "categorical";
                node["reference"] = reference;
            }
            output.push_back(node);

        }
        std::string result = output.dump(2);
        if(Configuration::verbose) { std::cout << "Storing Metadata in: " << Configuration::datatset_encoding << std::endl; }
        std::ofstream out(Configuration::datatset_encoding);
        out << result;
        out.close();
        return;
    }


    std::unordered_set< Model > models;
    // Extraction of Rashomon Set 
    if (Configuration::rashomon) { 
        float rashomon_bound;
        if (Configuration::rashomon_bound != 0) {
            rashomon_bound = Configuration::rashomon_bound;
        } else {
            if (Configuration::verbose) {
                std::cout << "Finding Optimal Objective..." << std::endl;
            }
            fit_gosdt(optimizer, models);

            float optimal_objective = models.begin() -> loss() + models.begin() -> complexity();
            if (Configuration::verbose) {
                std::cout << "Found Optimal Objective: " << optimal_objective << std::endl;
            }
            
            if (Configuration::rashomon_bound_multiplier != 0) {
                rashomon_bound = optimal_objective * (1 + Configuration::rashomon_bound_multiplier);
            } else {
                rashomon_bound = optimal_objective + Configuration::rashomon_bound_adder;
            }
        }
        fit_rashomon(optimizer, rashomon_bound, results);
        process_rashomon_result(results);
    } else {
        fit_gosdt(optimizer, models);
        float optimal_objective = models.begin() -> loss() + models.begin() -> complexity();
        if (Configuration::verbose) {
            std::cout << "Found Optimal Objective: " << optimal_objective << std::endl;
        }
    }
}

void GOSDT::fit_gosdt(Optimizer & optimizer, std::unordered_set< Model > & models) {
    GOSDT::time = 0.0;
    GOSDT::size = 0;
    GOSDT::iterations = 0;
    GOSDT::status = 0;

    std::vector< std::thread > workers;
    std::vector< int > iterations(Configuration::worker_limit);

    if(Configuration::verbose) { std::cout << "Starting Search for the Optimal Solution" << std::endl; }
    auto start = std::chrono::high_resolution_clock::now();

    optimizer.initialize();
    for (unsigned int i = 0; i < Configuration::worker_limit; ++i) {
        workers.emplace_back(work, i, std::ref(optimizer), std::ref(iterations[i]));
        #if  !defined(__APPLE__) && !defined(_WIN32)
        if (Configuration::worker_limit > 1) {
            // If using Ubuntu Build, we can pin each thread to a specific CPU core to improve cache locality
            cpu_set_t cpuset; CPU_ZERO(&cpuset); CPU_SET(i, &cpuset);
            int error = pthread_setaffinity_np(workers[i].native_handle(), sizeof(cpu_set_t), &cpuset);
            if (error != 0) { std::cerr << "Error calling pthread_setaffinity_np: " << error << std::endl; }
        }
        #endif
    }
    for (auto iterator = workers.begin(); iterator != workers.end(); ++iterator) { (* iterator).join(); } // Wait for the thread pool to terminate
    
    auto stop = std::chrono::high_resolution_clock::now(); // Stop measuring training time
    GOSDT::time = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count() / 1000.0;
    if(Configuration::verbose) { std::cout << "Optimal Solution Search Complete" << std::endl; }

    for (auto iterator = iterations.begin(); iterator != iterations.end(); ++iterator) { GOSDT::iterations += * iterator; }    
    GOSDT::size = optimizer.size();

    if (Configuration::timing != "") {
        std::ofstream timing_output(Configuration::timing, std::ios_base::app);
        timing_output << GOSDT::time;
        timing_output.flush();
        timing_output.close();
    }

    if(Configuration::verbose) {
        std::cout << "Training Duration: " << GOSDT::time << " seconds" << std::endl;
        std::cout << "Number of Iterations: " << GOSDT::iterations << " iterations" << std::endl;
        std::cout << "Size of Graph: " << GOSDT::size << " nodes" << std::endl;
        float lowerbound, upperbound;
        optimizer.objective_boundary(& lowerbound, & upperbound);
        std::cout << "Objective Boundary: [" << lowerbound << ", " << upperbound << "]" << std::endl;
        std::cout << "Optimality Gap: " << optimizer.uncertainty() << std::endl;
    }

    // try 
    { // Model Extraction
        if (!optimizer.complete()) {
            GOSDT::status = 1;
            if (Configuration::diagnostics) {
                std::cout << "Non-convergence Detected. Beginning Diagnosis" << std::endl;
                optimizer.diagnose_non_convergence();
                std::cout << "Diagnosis complete" << std::endl;
            }
        }
        optimizer.models(models);

        if (Configuration::model_limit > 0 && models.size() == 0) {
            GOSDT::status = 1;
            if (Configuration::diagnostics) {
                std::cout << "False-convergence Detected. Beginning Diagnosis" << std::endl;
                optimizer.diagnose_false_convergence();
                std::cout << "Diagnosis complete" << std::endl;
            }
        }

        if (Configuration::verbose) {
            std::cout << "Models Generated: " << models.size() << std::endl;
            if (optimizer.uncertainty() == 0.0 && models.size() > 0) {
                std::cout << "Loss: " << models.begin() -> loss() << std::endl;
                std::cout << "Complexity: " << models.begin() -> complexity() << std::endl;
            } 
        }
        if (Configuration::model != "") {
            json output = json::array();
            for (auto iterator = models.begin(); iterator != models.end(); ++iterator) {
                Model model = * iterator;
                json object = json::object();
                model.to_json(object);
                output.push_back(object);
            }
            std::string result = output.dump(2);
            if(Configuration::verbose) { std::cout << "Storing Models in: " << Configuration::model << std::endl; }
            std::ofstream out(Configuration::model);
            out << result;
            out.close();
        }

    }
    optimizer.reset_except_dataset();
}

void GOSDT::fit_rashomon(Optimizer & optimizer, float rashomon_bound, results_t & results) {
    GOSDT::time = 0.0;
    GOSDT::size = 0;
    GOSDT::iterations = 0;
    GOSDT::status = 0;
    std::vector< std::thread > workers;
    std::vector< int > iterations(Configuration::worker_limit);

    if(Configuration::verbose) { std::cout << "Starting Extraction of Rashomon Set" << std::endl; }
    auto start = std::chrono::high_resolution_clock::now();

    optimizer.initialize();
    if (Configuration::verbose) {
        std::cout << "Using Rashomon bound: " << rashomon_bound << std::endl;
    }
    optimizer.set_rashomon_bound(rashomon_bound);
    optimizer.set_rashomon_flag();
    for (unsigned int i = 0; i < Configuration::worker_limit; ++i) {
        workers.emplace_back(work, i, std::ref(optimizer), std::ref(iterations[i]));
        #if  !defined(__APPLE__) && !defined(_WIN32)
            if (Configuration::worker_limit > 1) {
            // If using Ubuntu Build, we can pin each thread to a specific CPU core to improve cache locality
            cpu_set_t cpuset; CPU_ZERO(&cpuset); CPU_SET(i, &cpuset);
            int error = pthread_setaffinity_np(workers[i].native_handle(), sizeof(cpu_set_t), &cpuset);
            if (error != 0) { std::cerr << "Error calling pthread_setaffinity_np: " << error << std::endl; }
        }
        #endif
    }
    for (auto iterator = workers.begin(); iterator != workers.end(); ++iterator) { (* iterator).join(); } // Wait for the thread pool to terminate
    
    auto stop = std::chrono::high_resolution_clock::now(); // Stop measuring training time
    GOSDT::time = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count() / 1000.0;
    if(Configuration::verbose) { std::cout << "Rashomon Set Construction Completed" << std::endl; }

    for (auto iterator = iterations.begin(); iterator != iterations.end(); ++iterator) { GOSDT::iterations += * iterator; }    
    GOSDT::size = optimizer.size();

    if (Configuration::timing != "") {
        std::ofstream timing_output(Configuration::timing, std::ios_base::app);
        timing_output << GOSDT::time;
        timing_output.flush();
        timing_output.close();
    }

    if(Configuration::verbose) {
        std::cout << "Training Duration: " << GOSDT::time << " seconds" << std::endl;
        std::cout << "Number of Iterations: " << GOSDT::iterations << " iterations" << std::endl;
        std::cout << "Size of Graph: " << GOSDT::size << " nodes" << std::endl;
        float lowerbound, upperbound;
        optimizer.objective_boundary(& lowerbound, & upperbound);
        std::cout << "Objective Boundary: [" << lowerbound << ", " << upperbound << "]" << std::endl;
        std::cout << "Optimality Gap: " << optimizer.uncertainty() << std::endl;
    }

    tic("Extraction Duration: ");
    optimizer.rash_models(results);
    toc();

    if (Configuration::verbose) {
        std::cout << "Stored keys size: " << results.first.size() << std::endl;
        // boost::multiprecision::uint128_t models_count = 0;
        long long unsigned int models_count = 0;
        for (auto model_set : results.second) {
            models_count += model_set.second->get_stored_model_count();
        }
        std::cout << "Size of Rashomon Set: " << models_count << std::endl;
        std::cout << "Memory usage after extraction: " << getCurrentRSS() / 1000000 << std::endl;
    }

    optimizer.reset_except_dataset();
}

void GOSDT::process_rashomon_result(results_t &results) {

    if (Configuration::output_accuracy_model_set) {
        tic("Output of accuracy Rashomon Set in Model Set: ");

        std::string serialization;
        ModelSet::serialize(results, serialization, 0);
        
        std::string file_name = "model_set-accuracy-" + Configuration::rashomon_model_set_suffix;

        if(Configuration::verbose) { std::cout << "Storing Models in: " << file_name << std::endl; }
        std::ofstream out(file_name);
        out << serialization;
        out.close();
        toc();
    }

    if (Configuration::output_covered_sets.size() != 0) {

        tic("Construction of (TP, TN, #Leaves) Duration: ");
        for (auto obj : results.first) {
            results.second[obj]->get_values_of_interest_count();
        }
        toc();

        unsigned int P, N;
        State::dataset.get_total_P_N(P, N);

        for (int i = 0; i < Configuration::output_covered_sets.size(); i++) {

            // Common data for this task
            CoveredSetExtraction covered_sets_type = Configuration::output_covered_sets[i];
            std::string covered_sets_type_string = Configuration::covered_set_type_to_string(covered_sets_type);
            double limit = Configuration::covered_sets_thresholds[i];

            tic("Extraction of " + covered_sets_type_string + " Rashomon Set: ");


            values_of_interest_mapping_t mapping_all;

            unsigned long long model_count = 0;

            for (auto obj : results.first) {
                auto output = results.second[obj]->get_values_of_interest_count();
                bool extract = false;
                for (auto i : output) {
                    auto values_of_interest = i.first;

                    double TP = values_of_interest.TP;
                    double TN = values_of_interest.TN;
                    auto reg = values_of_interest.regularization;

                    double metric = Configuration::computeScore(covered_sets_type, P, N, TP, TN);
                    double obj_value = 1 - metric + Configuration::regularization * reg;

                    if (obj_value <= limit) {
                        model_count += i.second;
                        extract = true;
                    }
                }
                // Here as I believe `get_values_of_interest_mapping` is a more costly process
                if (extract) {
                    auto mapping = results.second[obj]->get_values_of_interest_mapping();
                    for (auto i : mapping) {
                        auto values_of_interest = i.first;

                        double TP = values_of_interest.TP;
                        double TN = values_of_interest.TN;
                        auto reg = values_of_interest.regularization;

                     
                        double metric = Configuration::computeScore(covered_sets_type, P, N, TP, TN);
                        double obj_value = 1 - metric + Configuration::regularization * reg;
                        
                        // assert(output.at(values_of_interest) == i.second->get_stored_model_count());

                        if (obj_value <= limit) {
                            auto existing_model_set = mapping_all.find(values_of_interest);
                            if (existing_model_set == mapping_all.end()) {
                                mapping_all.insert(i);
                            } else {
                                existing_model_set->second->merge(i.second);
                            }
                        }
                    }
                }
            }
            std::cout << "Size of " + covered_sets_type_string + " Rashomon Set: " << model_count << std::endl;

            toc();

            tic("Output of " + covered_sets_type_string + " Rashomon Set in Model Set: ");

            std::string serialization;
            ModelSet::serialize(mapping_all, serialization, 0);
            
            std::string file_name = "model_set-" + covered_sets_type_string + "-" + Configuration::rashomon_model_set_suffix;

            if(Configuration::verbose) { std::cout << "Storing Models in: " << file_name << std::endl; }
            std::ofstream out(file_name);
            out << serialization;
            out.close();

            toc();
        }
    }
    
    if (Configuration::rashomon_trie != "") {

        tic("Insertion of Rashomon Set into Trie: ");

        bool calculate_size = false;
        char const *type = "node";
        Trie* tree = new Trie(calculate_size, type);
        tree->insert_root();

        for (auto obj : results.first) {
            tree->insert_model_set(results.second[obj]);
        }
        
        toc();

        tic("Output of Rashomon Set in Trie: ");

        std::string serialization;
        tree->serialize(serialization, 0);

        if (Configuration::verbose) {
            std::cout << "Storing Models in: " << Configuration::rashomon_trie << std::endl;
        }
        std::ofstream out(Configuration::rashomon_trie);
        out << serialization;
        out.close();

        toc();
    }
}

void GOSDT::work(int const id, Optimizer & optimizer, int & return_reference) {
    unsigned int iterations = 0;
    try {
        while (optimizer.iterate(id)) { iterations += 1; }
    } catch( IntegrityViolation exception ) {
        GOSDT::status = 1;
        std::cout << exception.to_string() << std::endl;
        throw std::move(exception);
    }
    return_reference = iterations;
}