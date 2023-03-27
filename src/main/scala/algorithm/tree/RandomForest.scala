package it.unibo.andrp
package algorithm.tree

import config.AlgorithmConfig

class RandomForest() {
    private var trees = new Array[DecisionTree](AlgorithmConfig.RandomForest.NUM_TREES)

    def getTrees: Array[DecisionTree] =
        trees

    def setTrees(trees: Array[DecisionTree]): Unit =
        this.trees = trees
}
