package it.unibo.andrp
package algorithm.tree

class DecisionTree(node: Option[Node] = None) {

    private var root = node

    inline def setRoot(node: Option[Node]): Unit =
        root = node
}

