package it.unibo.andrp
package algorithm.tree

class DecisionTree(node: Option[Node] = None) {

    private var root = node

    def setRoot(node: Option[Node]): Unit =
        root = node

    def getRoot: Option[Node] =
        root
}

