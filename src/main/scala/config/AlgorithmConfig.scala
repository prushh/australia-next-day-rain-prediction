package it.unibo.andrp
package config

import algorithm.knn.Distances

object AlgorithmConfig {

    val TRAINING_RATIO = 0.8
    val SEED = 42L

    object Knn {
        val NUMBER_NEAREST_NEIGHBORS = 3
        val DISTANCE_METHOD = Distances.Euclidean
    }
}
