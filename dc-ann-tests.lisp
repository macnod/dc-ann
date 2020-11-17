(in-package :cl-user)
(require :dc-ann)
(require :prove)
(defpackage dc-ann-tests (:use :cl :prove))
(in-package :dc-ann-tests)

(plan 1)

(ok (loop with expected-outputs = (list 0.7310586 0.26894143 1.0 0.0 0.5 0.5
                                        0.880797 0.95257413 0.98201376 0.9933072
                                        0.9975274 0.999089 0.99966466 0.9998766
                                        0.9999546 0.11920292 0.047425874
                                        0.01798621 0.006692851 0.002472623
                                        9.110512e-4 3.3535014e-4 1.2339458e-4
                                        4.5397872e-5) 
       for input in (list 1 -1 1e10 -1e10 1e-10 -1e-10 
                          2  3  4  5  6  7  8  9  10 
                         -2 -3 -4 -5 -6 -7 -8 -9 -10)
       for expected-output in expected-outputs
       for output = (dc-ann::logistic input)
       always (< (abs (- output expected-output)) 1e-10)))

(finalize)
