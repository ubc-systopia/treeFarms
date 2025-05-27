#include "configuration.hpp"

float Configuration::uncertainty_tolerance = 0.0;
float Configuration::regularization = 0.05;
float Configuration::upperbound = 0.0;

unsigned int Configuration::time_limit = 0;
unsigned int Configuration::worker_limit = 1;
unsigned int Configuration::stack_limit = 0;
unsigned int Configuration::precision_limit = 0;
unsigned int Configuration::model_limit = 10000;

bool Configuration::verbose = false;
bool Configuration::diagnostics = false;

unsigned char Configuration::depth_budget = 0;

unsigned int Configuration::minimum_captured_points = 0;

std::vector<int> Configuration::memory_checkpoints = {}; 

bool Configuration::output_accuracy_model_set = false; 
std::vector<CoveredSetExtraction> Configuration::output_covered_sets = {}; 
std::vector<double> Configuration::covered_sets_thresholds = {};

bool Configuration::balance = false;
bool Configuration::look_ahead = true;
bool Configuration::similar_support = false;
bool Configuration::cancellation = true;
bool Configuration::continuous_feature_exchange = false;
bool Configuration::feature_exchange = false;
bool Configuration::feature_transform = true;
bool Configuration::rule_list = false;
bool Configuration::non_binary = false;

std::string Configuration::costs = "";
std::string Configuration::model = "";
std::string Configuration::rashomon_model = "";
std::string Configuration::rashomon_model_set_suffix = "";
std::string Configuration::rashomon_trie = "";
std::string Configuration::timing = "";
std::string Configuration::trace = "";
std::string Configuration::tree = "";
std::string Configuration::profile = "";
std::string Configuration::datatset_encoding = "";

bool Configuration::rashomon = true;  
// Below we default the multiplier to 0.05 when no rashomon bound is specified in the configuration file
float Configuration::rashomon_bound = 0.0; // 
float Configuration::rashomon_bound_multiplier = 0.0; // 
float Configuration::rashomon_bound_adder = 0.0; // 
bool Configuration::rashomon_ignore_trivial_extensions = true;


void Configuration::configure(std::istream & source) {
    json config;
    source >> config;
    Configuration::configure(config);
};

void Configuration::configure(json config) {
    if (config.contains("uncertainty_tolerance")) { Configuration::uncertainty_tolerance = config["uncertainty_tolerance"]; }
    if (config.contains("regularization")) { Configuration::regularization = config["regularization"]; }
    if (config.contains("upperbound")) { Configuration::upperbound = config["upperbound"]; }

    if (config.contains("time_limit")) { Configuration::time_limit = config["time_limit"]; }
    if (config.contains("worker_limit")) { Configuration::worker_limit = config["worker_limit"]; }
    if (config.contains("stack_limit")) { Configuration::stack_limit = config["stack_limit"]; }
    if (config.contains("precision_limit")) { Configuration::precision_limit = config["precision_limit"]; }
    if (config.contains("model_limit")) { Configuration::model_limit = config["model_limit"]; }

    if (config.contains("verbose")) { Configuration::verbose = config["verbose"]; }
    if (config.contains("diagnostics")) { Configuration::diagnostics = config["diagnostics"]; }

    if (config.contains("depth_budget")) { Configuration::depth_budget = config["depth_budget"]; }

    if (config.contains("minimum_captured_points")) { Configuration::minimum_captured_points = config["minimum_captured_points"]; }

    if (config.contains("memory_checkpoints")) { Configuration::memory_checkpoints = config["memory_checkpoints"].get<std::vector<int>>(); }
    
    if (config.contains("output_accuracy_model_set")) { Configuration::output_accuracy_model_set = config["output_accuracy_model_set"]; }
    if (config.contains("output_covered_sets")) { 
        std::vector<CoveredSetExtraction> types = {};
        for (auto i : config["output_covered_sets"].get<std::vector<std::string>>()) {
            if (i == "f1") {
                types.emplace_back(F1);
            } else if (i == "bacc") {
                types.emplace_back(BACC);
            } else if (i == "auc") {
                types.emplace_back(AUC);
            } else {
                throw std::invalid_argument("Wrong arguments");
            }
        }
        Configuration::output_covered_sets = types; }
    if (config.contains("covered_sets_thresholds")) { 
        Configuration::covered_sets_thresholds = config["covered_sets_thresholds"].get<std::vector<double>>(); 
        assert(Configuration::covered_sets_thresholds.size() == Configuration::output_covered_sets.size());
    }

    if (config.contains("balance")) { Configuration::balance = config["balance"]; }
    if (config.contains("look_ahead")) { Configuration::look_ahead = config["look_ahead"]; }
    if (config.contains("similar_support")) { Configuration::similar_support = config["similar_support"]; }
    if (config.contains("cancellation")) { Configuration::cancellation = config["cancellation"]; }
    if (config.contains("continuous_feature_exchange")) { Configuration::continuous_feature_exchange = config["continuous_feature_exchange"]; }
    if (config.contains("feature_exchange")) { Configuration::feature_exchange = config["feature_exchange"]; }
    if (config.contains("feature_transform")) { Configuration::feature_transform = config["feature_transform"]; }
    if (config.contains("rule_list")) { Configuration::rule_list = config["rule_list"]; }
    if (config.contains("non_binary")) { Configuration::non_binary = config["non_binary"]; }

    if (config.contains("costs")) { Configuration::costs = config["costs"].get<std::string>();; }
    if (config.contains("model")) { Configuration::model = config["model"].get<std::string>();; }
    if (config.contains("rashomon_model")) { Configuration::rashomon_model = config["rashomon_model"].get<std::string>();; }
    if (config.contains("rashomon_model_set_suffix")) { Configuration::rashomon_model_set_suffix = config["rashomon_model_set_suffix"].get<std::string>();; }
    if (config.contains("rashomon_trie")) { Configuration::rashomon_trie = config["rashomon_trie"].get<std::string>();; }
    if (config.contains("timing")) { Configuration::timing = config["timing"].get<std::string>();; }
    if (config.contains("trace")) { Configuration::trace = config["trace"].get<std::string>();; }
    if (config.contains("tree")) { Configuration::tree = config["tree"].get<std::string>();; }
    if (config.contains("profile")) { Configuration::profile = config["profile"].get<std::string>();; }
    if (config.contains("datatset_encoding")) { Configuration::datatset_encoding = config["datatset_encoding"].get<std::string>();; }

    if (config.contains("rashomon")) { Configuration::rashomon = config["rashomon"]; }
    if (config.contains("rashomon_bound")) { Configuration::rashomon_bound = config["rashomon_bound"]; }
    if (config.contains("rashomon_bound_multiplier")) { Configuration::rashomon_bound_multiplier = config["rashomon_bound_multiplier"]; }
    if (config.contains("rashomon_bound_adder")) { Configuration::rashomon_bound_adder = config["rashomon_bound_adder"]; }
    if (!config.contains("rashomon_bound") && !config.contains("rashomon_bound_multiplier") && !config.contains("rashomon_bound_adder")) {
        Configuration::rashomon_bound_multiplier = 0.05;  // defaults the multiplier to 0.05 when no rashomon bound is specified
    }
    if (Configuration::rashomon) {
        assert(int(Configuration::rashomon_bound == 0) +
                   int(Configuration::rashomon_bound_multiplier == 0) +
                   int(Configuration::rashomon_bound_adder == 0) ==
               2);
    }
    if (config.contains("rashomon_ignore_trivial_extensions")) { Configuration::rashomon_ignore_trivial_extensions = config["rashomon_ignore_trivial_extensions"]; }


}

std::string Configuration::to_string(unsigned int spacing) {
    json obj = json::object();
    obj["uncertainty_tolerance"] = Configuration::uncertainty_tolerance;
    obj["regularization"] = Configuration::regularization;
    obj["upperbound"] = Configuration::upperbound;

    obj["time_limit"] = Configuration::time_limit;
    obj["worker_limit"] = Configuration::worker_limit;
    obj["stack_limit"] = Configuration::stack_limit;
    obj["precision_limit"] = Configuration::precision_limit;
    obj["model_limit"] = Configuration::model_limit;

    obj["verbose"] = Configuration::verbose;
    obj["diagnostics"] = Configuration::diagnostics;

    obj["depth_budget"] = Configuration::depth_budget;

    obj["minimum_captured_points"] = Configuration::minimum_captured_points;

    obj["memory_checkpoints"] = Configuration::memory_checkpoints;

    obj["output_accuracy_model_set"] = Configuration::output_accuracy_model_set;
    obj["output_covered_sets"] = Configuration::output_covered_sets;
    obj["covered_sets_thresholds"] = Configuration::covered_sets_thresholds;

    obj["balance"] = Configuration::balance;
    obj["look_ahead"] = Configuration::look_ahead;
    obj["similar_support"] = Configuration::similar_support;
    obj["cancellation"] = Configuration::cancellation;
    obj["continuous_feature_exchange"] = Configuration::continuous_feature_exchange;
    obj["feature_exchange"] = Configuration::feature_exchange;
    obj["feature_transform"] = Configuration::feature_transform;
    obj["rule_list"] = Configuration::rule_list;
    obj["non_binary"] = Configuration::non_binary;

    obj["costs"] = Configuration::costs;
    obj["model"] = Configuration::model;
    obj["rashomon_model"] = Configuration::rashomon_model;
    obj["rashomon_model_set_suffix"] = Configuration::rashomon_model_set_suffix;
    obj["rashomon_trie"] = Configuration::rashomon_trie;
    obj["timing"] = Configuration::timing;
    obj["trace"] = Configuration::trace;
    obj["tree"] = Configuration::tree;
    obj["profile"] = Configuration::profile;
    obj["datatset_encoding"] = Configuration::datatset_encoding;

    obj["rashomon"] = Configuration::rashomon;
    obj["rashomon_bound"] = Configuration::rashomon_bound;
    obj["rashomon_bound_multiplier"] = Configuration::rashomon_bound_multiplier;
    obj["rashomon_bound_adder"] = Configuration::rashomon_bound_adder;
    obj["rashomon_ignore_trivial_extensions"] = Configuration::rashomon_ignore_trivial_extensions;

    return obj.dump(spacing);
}
