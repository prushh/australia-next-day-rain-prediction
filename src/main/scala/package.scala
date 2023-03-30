package it.unibo


package object andrp {
    def time[R](label: String, block: => R): (R, Long) = {
        val t0 = System.nanoTime()
        val result = block
        val t1 = System.nanoTime()
        val time: Long = (t1 - t0) / 1000000
        println(s"$label - elapsed time: " + (t1 - t0) / 1000000 + "ms")
        (result, time)
    }
}
