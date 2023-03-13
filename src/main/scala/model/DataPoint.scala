package it.unibo.andrp
package model

case class DataPoint(features: Seq[Double], label: Double) extends Serializable
