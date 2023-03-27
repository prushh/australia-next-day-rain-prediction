package it.unibo.andrp
package algorithm.tree

import model.DataPoint

trait Node {
    def predict(dataPoint: DataPoint): Double
}