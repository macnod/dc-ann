(in-package :cl-user)
(require :dc-ann)
(require :prove)
(defpackage dc-ann-tests (:use :cl :prove))
(in-package :dc-ann-tests)

(defun round-to-decimal-count (x decimal-count)
  (float (/ (round (* x (expt 10 decimal-count))) (expt 10 decimal-count))))

(defun basic-net (id)
  (make-instance 'dc-ann::t-net 
                 :id id
                 :topology '((:count 2 :transfer :logistic)
                             (:count 4 :transfer :logistic)
                             (:count 1 :transfer :logistic))))

(plan 7)

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

(ok (loop for tests from 1 to 100
       for input = (- (random 1e6) 5e5)
       for output = (dc-ann::relu input)
       for leaky-output = (dc-ann::relu-leaky input)
       always (if (< input 0) (zerop output) (= input output)))
    "Relu returns sane values.")

(ok (loop for (input expected-output) in
         '((0.5025 0.25) (0.6425 0.2297) (0.5102 0.2499) (0.6417 0.2299) 
           (0.0119 0.0118) (0.8801 0.1055) (0.3982 0.2396) (0.8253 0.1442) 
           (0.7818 0.1706) (0.8463 0.1301) (0.0 0.0) (0.5 0.25) (1.0 0.0))
       for output = (round-to-decimal-count (dc-ann::logistic-derivative input) 4)
       always (= output expected-output))
    "Logistic function derivative returns expected values.")

(ok (loop for tests from 1 to 100
       for input = (- (random 1e6) 5e5)
       for output = (dc-ann::relu-derivative input)
       for leaky-output = (dc-ann::relu-leaky-derivative input)
       always (if (<= input 0) 
                  (and (zerop output) (= leaky-output 0.001))
                  (and (= output 1) (= leaky-output 1))))
    "Relu and relu-leaky function derivatives return expected values.")

(ok (loop for name in '(:logistic :relu :relu-leaky)
       for expected-output in '(0.7311 1 1)
       for transfer = (dc-ann::transfer (gethash name dc-ann::*transfer-functions*))
       for output = (round-to-decimal-count (funcall transfer 1.0) 4)
       always (= output expected-output))
    "*transfer-functions* hash table correctly loaded.")

(finalize)
