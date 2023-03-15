package it.unibo.andrp
package config

import algorithm.knn.Distance

object AlgorithmConfig {

    val TRAINING_RATIO = 0.8
    val SEED = 42L

    object Knn {
        val NUMBER_NEAREST_NEIGHBORS = 3
        val DISTANCE_METHOD = Distance.Euclidean
    }
}
