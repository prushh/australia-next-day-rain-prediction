package it.unibo.andrp
package algorithm.tree

import model.DataPoint

case class Leaf(label: Int) extends Node {
    def predict(dataPoint: DataPoint): Int = label
}
