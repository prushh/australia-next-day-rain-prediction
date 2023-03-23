package it.unibo.andrp
package algorithm.tree

import model.DataPoint

case class InternalNode(feature: Int, threshold: Double, left: Node, right: Node) extends Node {
    def predict(dataPoint: DataPoint): Int =
        if (dataPoint.features(feature) <= threshold) left.predict(dataPoint) else right.predict(dataPoint)
}
