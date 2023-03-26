package it.unibo.andrp
package model

abstract class Point(humidity3: Double,
                     rainToday: Int,
                     cloud3: Double,
                     humidity9: Double,
                     cloud9: Double,
                     rainfall: Double,
                     windSpeed: Double,
                     windSpeed9: Double,
                     windSpeed3: Double,
                     minTemp: Double,
                     windDir: Double,
                     windDir9: Double,
                     windDir3: Double,
                     rainTomorrow: Int) extends Serializable