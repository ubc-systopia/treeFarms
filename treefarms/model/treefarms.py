import json
import pandas as pd
import time
from numpy import array
from sklearn.metrics import confusion_matrix, accuracy_score
import timbertrek

import treefarms.libgosdt as treefarms # Import the TREEFARMS extension ()
from treefarms.model.encoder import Encoder
from treefarms.model.imbalance.osdt_imb_v9 import bbound, predict # Import the special objective implementation
from treefarms.model.tree_classifier import TreeClassifier # Import the tree classification model
from treefarms.model.model_set import ModelSetContainer

class TREEFARMS:
    def __init__(self, configuration={}):
        self.configuration = configuration
        self.time = 0.0
        self.stime = 0.0
        self.utime = 0.0
        self.maxmem = 0
        self.numswap = 0
        self.numctxtswitch = 0
        self.iterations = 0
        self.size = 0
        self.tree = None
        self.encoder = None
        self.lb = 0
        self.ub = 0
        self.timeout = False
        self.reported_loss = 0
        self.model_set = None
        self.dataset = None
        self.verbose = False if 'verbose' not in configuration.keys() else configuration['verbose']

    # TODO: implement this
    def load(self, path):
        """
        Parameters
        ---
        path : string
            path to a JSON file representing a model
        """
        raise NotImplementedError

    def __train__(self, X, y):
        """
        Parameters
        ---
        X : matrix-like, shape = [n_samples by m_features]
            matrix containing the training samples and features
        y : array-like, shape = [n_samples by 1]
            column containing the correct label for each sample in X
        Modifies
        ---
        trains a model using the TREEFARMS native extension
        """
        (n, m) = X.shape
        dataset = X.copy()
        y_name = y.name
        if not y_name:
            y_name = "class"
        dataset.insert(m, y_name, y) # It is expected that the last column is the label column

        treefarms.configure(json.dumps(self.configuration, separators=(',', ':')))
        result = treefarms.fit(dataset.to_csv(index=False)) # Perform extension call to train the model

        self.time = treefarms.time() # Record the training time
        # self.stime = treefarms.stime()
        # self.utime = treefarms.utime()

        if treefarms.status() == 0:
            if self.verbose:
                print("treefarms reported successful execution")
            self.timeout = False
        elif treefarms.status() == 2:
            print("treefarms reported possible timeout.")
            self.timeout = True
            self.time = -1
            self.stime = -1
            self.utime = -1
        else :
            print('----------------------------------------------')
            print(result)
            print('----------------------------------------------')
            raise Exception("Error: TREEFARMS encountered an error while training")

        result = json.loads(result) # Deserialize result
        self.dataset = dataset
        self.model_set = ModelSetContainer(result)

        if self.verbose:
            print(f"training completed. Number of trees in the Rashomon set: {self.model_set.get_tree_count()}")


    def fit(self, X, y):
        """
        Parameters
        ---
        X : matrix-like, shape = [n_samples by m_features]
            matrix containing the training samples and features
        y : array-like, shape = [n_samples by 1]
            column containing the correct label for each sample in X
        Modifies
        ---
        trains the model so that this model instance is ready for prediction
        """
        self.__train__(X, y)
        return self

    # TODO: implement this
    def predict(self, X):
        """
        Parameters
        ---
        X : matrix-like, shape = [n_samples by m_features]
            a matrix where each row is a sample to be predicted and each column is a feature to be used for prediction
        Returns
        ---
        array-like, shape = [n_sampels by 1] : a column where each element is the prediction associated with each row
        """
        raise NotImplementedError

    def __getitem__(self, idx):
        """Obtain the `idx`th tree in the Rashomon set

        Parameters
        ----------
        idx : int
            Index of the tree in the Rashomon set

        Returns
        -------
        TreeClassifier
            A tree classifier at index `idx`
        """
        if self.model_set is None:
            raise Exception("Error: Model not yet trained")
        return self.model_set.__getitem__(idx)

    def get_tree_count(self):
        """Returns the number of trees in the Rashomon set

        Returns
        -------
        int
            Number of trees in the Rashomon set
        """
        if self.model_set is None:
            raise Exception("Error: Model not yet trained")
        return self.model_set.get_tree_count()

    def get_decision_paths(self, feature_names=None, feature_description=None):
        """Create a hierarchical dictionary describing the decision paths in the
        Rashomon set using `timbertrek`.
        Parameters
        ---
        feature_names : matrix-like, shape = [m_features + 1]
            a matrix where each row is a sample to be predicted and each column is a feature to be used for prediction
        """
        if self.model_set is None:
            raise Exception("Error: Model not yet trained")

        # Convert the trie structure to decision paths
        trie = self.model_set.to_trie()
        df = self.dataset
        if feature_names is None:
            feature_names = df.columns

        decision_paths = timbertrek.transform_trie_to_rules(
            trie,
            df,
            feature_names=feature_names,
            feature_description=feature_description,
        )

        return decision_paths

    def visualize(self, feature_names=None, feature_description=None, *, width=500, height=650):
        """Generates a visualization of the Rashomon set using `timbertrek`
        Parameters
        ---
        feature_names : matrix-like, shape = [m_features + 1]
            a matrix where each row is a sample to be predicted and each column is a feature to be used for prediction
        """
        # Get the decision paths
        decision_paths = self.get_decision_paths(
            feature_names=feature_names,
            feature_description=feature_description
        )

        # Show in the in-notebook visualization
        timbertrek.visualize(decision_paths, width=width, height=height)

    def __translate__(self, leaves):
        """
        Converts the leaves of OSDT into a TreeClassifier-compatible object
        """
        if len(leaves) == 1:
            return {
                "complexity": self.configuration["regularization"],
                "loss": 0,
                "name": "class",
                "prediction": list(leaves.values())[0]
            }
        else:
            features = {}
            for leaf in leaves.keys():
                if not leaf in features:
                    for e in leaf:
                        features[abs(e)] = 1
                    else:
                        features[abs(e)] += 1
            split = None
            max_freq = 0
            for feature, frequency in features.items():
                if frequency > max_freq:
                    max_freq = frequency
                    split = feature
            positive_leaves = {}
            negative_leaves = {}
            for leaf, prediction in leaves.items():
                if split in leaf:
                    positive_leaves[tuple(s for s in leaf if s != split)] = prediction
                else:
                    negative_leaves[tuple(s for s in leaf if s != -split)] = prediction
            return {
                "feature": split,
                "name": "feature_" + str(split),
                "reference": 1,
                "relation": "==",
                "true": self.__translate__(positive_leaves),
                "false": self.__translate__(negative_leaves),
            }
