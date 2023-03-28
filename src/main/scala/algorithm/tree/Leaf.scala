package it.unibo.andrp
package algorithm.tree

import model.DataPoint

case class Leaf(label: Double) extends Node {
    def predict(dataPoint: DataPoint): Double = label
}
