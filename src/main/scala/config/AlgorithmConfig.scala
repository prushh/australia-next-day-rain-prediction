package it.unibo.andrp
package config

import algorithm.knn.Distances
import algorithm.tree.{Impurities, SubsetStrategies}

object AlgorithmConfig {

    val TRAINING_RATIO = 0.8
    val SEED = 42L

    object Knn {
        val NUMBER_NEAREST_NEIGHBORS = 3
        val DISTANCE_METHOD = Distances.Euclidean
    }

    object DecisionTree {
        val MAX_DEPTH = 4
        val IMPURITY_MEASURE = Impurities.Gini
        val MIN_SPLIT_SIZE = 5
        val SUBSET_STRATEGY = SubsetStrategies.All
    }
    
    object RandomForest {
        val NUM_TREES = 5
    }
}
