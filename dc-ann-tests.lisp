(in-package :cl-user)
(require :dc-ann)
(require :prove)
(defpackage dc-ann-tests (:use :cl :prove))
(in-package :dc-ann-tests)

(defun round-to-decimal-count (x decimal-count)
  (float (/ (round (* x (expt 10 decimal-count))) (expt 10 decimal-count))))

(plan 3)

(ok (loop for (input expected-output) in
         '((1 0.7311) (-1 0.2689) (1.0e10 1.0) (-1.0e10 0.0)
           (1.0e20 1.0) (-1.0e20 0.0) (1.0e-10 0.5) (-1.0e-10 0.5)
           (2 0.8808) (3 0.9526) (4 0.9820) (5 0.9933)
           (6 0.9975) (7 0.9991) (8 0.9997) (9 0.9999)
           (10 1.0) (-2 0.1192) (-3 0.0474) (-4 0.0180)
           (-5 0.0067) (-6 0.0025) (-7 0.0009) (-8 0.0003)
           (-9 0.0001) (-10 0.0))
       for output = (round-to-decimal-count (dc-ann::logistic input) 4)
       always (= output expected-output))
    "Logistic function returns expected values for extreme inputs.")

(ok (loop with max = 1e9 and min = -1e9
       with range = (- max min)
       for tests from 1 to 100
       for input = (+ (random (float range)) min)
       for output = (dc-ann::logistic input)
       always (and (>= output 0.0)
                   (<= output 1.0)
                   (if (< input 0)
                       (<= output (- 1 output))
                       (>= output (- 1 output)))))
    "Logistic function near 0 for negative inputs; near 1 for positive inputs.")

(ok (loop for (input expected-output) in
         '((1 1.0) (-1 0.0) (1.0e10 1.0e10) (-1.0e10 0.0) 
           (1.0e20 1.0e20) (-1.0e20 0.0) (1.0e-10 0.0) (-1.0e-10 0.0) 
           (2 2.0) (3 3.0) (4 4.0) (5 5.0) (6 6.0) (7 7.0) (8 8.0) 
           (9 9.0) (10 10.0) (-2 0.0) (-3 0.0) (-4 0.0) (-5 0.0) 
           (-6 0.0) (-7 0.0) (-8 0.0) (-9 0.0) (-10 0.0))
         for output = (round-to-decimal-count (dc-ann::relu input) 4)
         for leaky-output = (round-to-decimal-count (dc-ann::relu-leaky input) 4)
         always (and (= output expected-output) 
                     (= leaky-output expected-output)))
    "Relu and relu-leaky functions return expected values for extreme inputs.")


(finalize)
