;; Copyright © 2002-2018 Donnie Cameron
;;
;; ANN stands for Artificial Neural Network. This is a simple
;; implementation of the standard backpropagation neural network.
;;

(in-package :dc-ann)

(defclass t-cx ()
  ((target :reader target :initarg :target :initform (error ":neuron required")
           :type t-neuron)
   (weight :accessor weight :initarg :weight :initform 1.0 :type real)
   (delta :accessor delta :initform 0.0 :type real))
  (:documentation "Describes a neural connection to TARGET neuron.  WEIGHT represents the strength of the connection and DELTA contains the last change in WEIGHT.  TARGET is required."))

(defclass t-neuron ()
  ((net :reader net :initarg :net :initform (error ":net required") :type t-net)
   (layer :reader layer :initarg :layer :initform (error ":layer required")
          :type integer)
   (layer-type :accessor layer-type :initarg :layer-type :initform nil)
   (biased :reader biased :initarg :biased :initform nil :type boolean)
   (id :accessor id :type integer)
   (index-in-layer :accessor index-in-layer :initarg :index-in-layer :type integer)
   (input :accessor input :initform 0.0 :type real)
   (transfer-tag :accessor transfer-tag :initarg :transfer-tag :initform :sigmoid)
   (transfer-function :accessor transfer-function :initform nil)
   (transfer-derivative :accessor transfer-derivative :initform nil)
   (output :accessor output :initform 0.0 :type real)
   (expected-output :accessor expected-output :initform 0.0 :type real)
   (err :accessor err :initform 0.0 :type real)
   (derivative :accessor derivative :initform 0.0 :type real)
   (x-coor :accessor x-coor :initform 0.0 :type real)
   (y-coor :accessor y-coor :initform 0.0 :type real)
   (label :accessor label :initform "" :initarg :label)
   (cxs :accessor cxs :initform nil :type list))
  (:documentation "Describes a neuron.  NET, required, is an object of type t-net that represents the neural network that this neuron is a part of.  LAYER, required, is an object of type t-layer that represents the neural network layer that this neuron belongs to.  If BIASED is true, the neuron will not have incoming connections.  ID is a distinct integer that identifies the neuron. X-COOR and Y-COOR allow this neuron to be placed in 2-dimentional space.  CXS contains the list of outgoing connections (of type t-cx) to other neurons."))

(defmethod initialize-instance :after ((neuron t-neuron) &key)
  (setf (id neuron) (incf (next-id (net neuron))))
  (unless (label neuron)
    (setf (label neuron) (format nil "~a" (id neuron))))
  (when (biased neuron)
    (setf (input neuron) 1.0))
  (setf (transfer-function neuron)
        (transfer-functions (transfer-tag neuron) :function))
  (setf (transfer-derivative neuron)
        (transfer-functions (transfer-tag neuron) :derivative)))

(defclass t-layer ()
  ((neuron-array :accessor neuron-array :type vector)
   (layer-index :reader layer-index :initarg :layer-index :initform
                (error ":layer-index required") :type integer)
   (layer-type :reader layer-type :initarg :layer-type :initform
               (error ":layer-type required") :type keyword)
   (neuron-count :accessor neuron-count :initarg :neuron-count
                 :initform (error ":neuron-count required") :type integer)
   (transfer-tag :accessor transfer-tag :initarg :transfer-tag
                 :initform nil)
   (next-layer :accessor next-layer :initarg :next-layer
               :type 't-layer :initform nil)
   (net :reader net :initarg :net :initform (error ":net required")
        :type t-net))
  (:documentation "Describes a neural network layer."))

(defmethod initialize-instance :after ((layer t-layer) &key)
  (let ((size (+ (neuron-count layer)
                 (if (equal (layer-type layer) :output) 0 1))))
    (setf (neuron-count layer) size)
    (setf (neuron-array layer)
          (make-array size :fill-pointer 0))
    (loop for a from 0 below size do
         (vector-push
          (make-instance 't-neuron
                         :layer (layer-index layer)
                         :layer-type (layer-type layer)
                         :index-in-layer a
                         :biased (and (not (equal (layer-type layer) :output))
                                      (= a (1- size)))
                         :net (net layer)
                         :transfer-tag (transfer-tag layer)
                         :label (when (and
                                       (equal (layer-type layer) :output)
                                       (output-labels (net layer)))
                                  (elt (output-labels (net layer)) a)))
          (neuron-array layer)))))

(defun bound-logistic (input)
  (let* ((limit 80)
         (bound-input (max (min input limit) (- limit))))
    (/ 1.0 (1+ (exp (- bound-input))))))

(defun bound-logistic-derivative (output)
  (* output (- 1 output)))

(defun logistic (input)
  (/ 1.0 (1+ (exp (- input)))))

(defun logistic-derivative (output)
  (* output (- 1 output)))

(defun relu (x)
  (max 0.0 x))

(defun relu-derivative (x)
  (if (>= x 0.0) 1.0 0.0))

(defun tanh-derivative (x) (- 1 (* x x)))

(defun identity-derivative (x)
  (declare (ignore x))
  1)

(defun no-activation (x) x)

(defun no-activation-derivative (x) x)

(defun transfer-functions (name function-or-derivative)
  (ds-get (ds `(:map :bound-logistic
                     (:map :function ,#'bound-logistic
                           :derivative ,#'bound-logistic-derivative)

                     :logistic
                     (:map :function ,#'logistic
                           :derivative ,#'logistic-derivative)

                     :sigmoid
                     (:map :function ,#'logistic
                           :derivative ,#'logistic-derivative)

                     :tanh
                     (:map :function ,#'tanh
                           :derivative ,#'tanh-derivative)

                     :identity
                     (:map :function ,#'identity
                           :derivative ,#'identity-derivative)

                     :relu
                     (:map :function ,#'relu
                           :derivative ,#'relu-derivative)

                     :rectified-linear
                     (:map :function ,#'relu
                           :derivative ,#'relu-derivative)

                     :none
                     (:map :function ,#'no-activation
                           :derivative ,#'no-activation-derivative)))
          name function-or-derivative))

(defclass t-net ()
  ((topology :reader topology :initarg :topology
             :initform (error ":topology required"))
   (output-labels :accessor output-labels :initarg :output-labels :initform nil)
   (learning-rate :accessor learning-rate :initarg :learning-rate :initform 0.3)
   (momentum :accessor momentum :initarg :momentum :initform 0.8)
   (wi :accessor wi :initform 0)
   (transfer-tag :accessor transfer-tag :initarg :transfer-tag :initform :logistic)
   (layers :accessor layers)
   (next-id :accessor next-id :initform 0)
   (min-mse :accessor min-mse :type real :initform 1000000.0)
   (max-mse :accessor max-mse :type real :initform -1000000.0)
   (mse-list :accessor mse-list :type list :initform nil)
   (id :reader id :initarg :id :type string :initform (unique-name))
   (log-file :accessor log-file :type string :initform nil)
   (last-anneal-iteration :accessor last-anneal-iteration
                          :type integer
                          :initform 0)
   (randomize :accessor randomize :type boolean :initform nil)
   (anneal :accessor anneal :type boolean :initform nil)
   (stop-training :accessor stop-training :type boolean :initform nil)
   (shock :accessor shock :type boolean :initform nil)
   (last-report-time :accessor last-report-time :type integer :initform nil))
  (:documentation "Describes a standard multilayer, fully-connected backpropagation neural network."))

(defmethod output-layer ((net t-net))
  (car (last (layers net))))

(defmethod outputs ((net t-net))
  (map 'vector 'output (neuron-array (output-layer net))))

(defmethod input-layer ((net t-net))
  (car (layers net)))

(defmethod get-neuron ((net t-net) (layer-index integer) (neuron-index integer))
  (elt (neuron-array (elt (layers net) layer-index)) neuron-index))

(defun set-layer-transfer-function (net layer-index transfer-tag)
  (loop for neuron across (neuron-array (elt (layers net) layer-index))
     do (setf (transfer-tag neuron) transfer-tag)
       (setf (transfer-function neuron)
             (transfer-functions transfer-tag :function))
       (setf (transfer-derivative neuron)
             (transfer-functions transfer-tag :derivative))))

(defun set-network-transfer-function (net transfer-tag)
  (loop for layer in (layers net) do
       (set-layer-transfer-function net (layer-index layer) transfer-tag)))

(defmethod transfer ((neuron t-neuron))
  (let ((transfer-function (transfer-function neuron)))
    (setf (output neuron)
          (funcall (transfer-function neuron) (input neuron)))
    neuron))

(defmethod connect ((net t-net))
  (loop for layer in (butlast (layers net)) do
       (loop for neuron across (neuron-array layer)
          for neuron-index = 1 then (1+ neuron-index)
          do (setf (cxs neuron)
                   (loop for target across
                        (neuron-array
                         (elt (layers net) (1+ (layer-index layer))))
                        for target-index = 1 then (1+ target-index)
                        when (not (biased target))
                        collect
                        (make-instance
                         't-cx
                         :target target
                         :weight (+ 0.5 (/ (sin (incf (wi net))) 2.0)))))))
  net)

(defmethod initialize-instance :after ((net t-net) &key)
  (setf (layers net)
        (loop with output-layer-index = (1- (length (topology net)))
           for layer-spec in (topology net)
           for layer-index = 0 then (1+ layer-index)
           collect
             (make-instance
              't-layer
              :layer-index layer-index
              :layer-type
              (cond ((zerop layer-index) :input)
                    ((= layer-index output-layer-index) :output)
                    (t :hidden))
              :neuron-count (elt (topology net) layer-index)
              :transfer-tag (transfer-tag net)
              :net net)))
  (loop for layer in (layers net)
     for next-layer = (if (eql (layer-type layer) :output)
                          nil
                          (elt (layers net) (1+ (layer-index layer))))
     do (setf (next-layer layer) next-layer))
  (connect net))

(defgeneric set-inputs (object inputs)
  (:method ((layer t-layer) (inputs list))
    (loop for neuron across (neuron-array layer)
       for input in inputs
       do (setf (input neuron) input)
       finally (return (map 'list
                            (lambda (x) (input x))
                            (neuron-array layer)))))
  (:method ((layer t-layer) (inputs vector))
    (loop for neuron across (neuron-array layer)
       for input across inputs
       do (setf (input neuron) input)
       finally (return (map 'list
                            (lambda (x) (input x))
                            (neuron-array layer)))))
  (:method ((net t-net) (inputs list))
    (loop for neuron across (neuron-array (car (layers net)))
       for input in inputs
       do (setf (input neuron) input)
       finally (return (map 'list
                            (lambda (x) (input x))
                            (neuron-array (car (layers net)))))))
  (:method ((net t-net) (inputs vector))
    (loop for neuron across (neuron-array (car (layers net)))
       for input across inputs
       do (setf (input neuron) input)
       finally (return (map 'list
                            (lambda (x) (input x))
                            (neuron-array (car (layers net))))))))

(defgeneric set-neuron-weights (net layer-index neuron-index weights)
  (:method ((net t-net) (layer-index integer) (neuron-index integer) (weights list))
    (let ((neuron (get-neuron net layer-index neuron-index)))
      (loop for cx in (cxs neuron)
         for weight in weights
         do (setf (weight cx) weight)
         finally (return (mapcar (lambda (x) (weight x)) (cxs neuron))))))
  (:method ((net t-net) (layer-index integer) (neuron-index integer) (weights array))
    (let ((neuron (get-neuron net layer-index neuron-index)))
      (loop for cx in (cxs neuron)
         for weight across weights
         do (setf (weight cx) weight)
         finally (return (mapcar (lambda (x) (weight x)) (cxs neuron)))))))

(defun layer-inputs (layer)
  (map 'list (lambda (x) (input x)) (neuron-array layer)))

(defun layer-outputs (layer)
  (map 'list (lambda (x) (output x)) (neuron-array layer)))

(defun net-input-for-neuron (net layer-index neuron-index)
  (loop with target-neuron = (get-neuron net layer-index neuron-index)
     for source-neuron across (neuron-array (elt (layers net) (1- layer-index)))
     for output = (output source-neuron)
     appending (loop for cx in (cxs source-neuron)
                  when (equal (target cx) target-neuron)
                  collect (list '* output (weight cx)))
     into terms-1
     appending (loop for cx in (cxs source-neuron)
                  when (equal (target cx) target-neuron)
                  collect (* output (weight cx)))
     into terms-2
     finally (return (list :terms terms-1 :net-input (apply '+ terms-2)))))

(defun set-neuron-input (net layer-index neuron-index input-value)
  (let ((neuron (get-neuron net layer-index neuron-index)))
    (setf (input neuron) input-value)))

(defun zero-all-neuron-inputs (net &optional exclude-biased-neurons)
  (loop for layer in (layers net) do
       (loop for neuron across (neuron-array layer)
          do (if exclude-biased-neurons
                 (when (not (biased neuron))
                   (setf (input neuron) 0.0))
                 (setf (input neuron) 0.0)))))

(defgeneric fire (object)
  (:method ((neuron t-neuron))
    (transfer neuron)
    (loop for cx in (cxs neuron)
       do (incf (input (target cx)) (* (weight cx) (output neuron))))
    neuron)
  (:method ((layer t-layer))
    (when (not (eql (layer-type layer) :output))
      (loop for neuron across (neuron-array (next-layer layer))
         do (setf (input neuron) 0.0)))
    (loop for neuron across (neuron-array layer) do (fire neuron))
    layer)
  (:method ((net t-net))
    (loop for layer in (layers net) do (fire layer))
    net))

(defgeneric feed (t-net inputs)
  (:method ((net t-net) (inputs list))
    (feed net (map 'vector 'identity inputs)))
  (:method ((net t-net) (inputs vector))
    ;; Copy input values to input layer
    (loop for input across inputs
       for neuron across (neuron-array (input-layer net))
       do (setf (input neuron) input))
    ;; Feed forward
    (fire net)
    ;; Return a vector with the output value of each output-layer neuron
    (map 'vector 'output (neuron-array (output-layer net)))))

(defgeneric winner (t-net inputs)
  (:method ((net t-net) (inputs list))
    (winner net (map 'vector 'identity inputs)))
  (:method ((net t-net) (inputs vector))
    (feed net inputs)
    (loop for neuron across (neuron-array (output-layer net))
       for top-neuron = neuron then (if (> (input neuron) (input top-neuron))
                                        neuron
                                        top-neuron)
       finally (return (label top-neuron)))))

;; Current
(defun backprop-output (neuron)
  (loop 
     for cx in (cxs neuron)
     for target-neuron = (target cx)
     do (setf (err target-neuron)
              (* (- (expected-output target-neuron)
                    (output target-neuron))
                 (funcall (transfer-derivative target-neuron)
                          (output target-neuron))))))

(defun backprop-hidden (neuron)
  (loop
     for cx in (cxs neuron)
     for target-neuron = (target cx)
     do (setf (err target-neuron)
              (* (loop for cx in (cxs target-neuron)
                    summing (* (weight cx) (err (target cx))))
                 (funcall (transfer-derivative target-neuron)
                          (output target-neuron))))))

(defgeneric backprop (component)
  (:method ((neuron t-neuron))
    (if (= (1+ (layer neuron)) (layer-index (output-layer (net neuron))))
        (backprop-output neuron)
        (backprop-hidden neuron))
    neuron)
  (:method ((layer t-layer))
    (loop for neuron across (neuron-array layer) do (backprop neuron))
    layer)
  (:method ((net t-net))
    (loop for layer in (reverse (butlast (layers net)))
       do (backprop layer))
    net))

(defgeneric update-weights (component)
  (:method ((neuron t-neuron))
    (loop
       with learning-rate = (learning-rate (net neuron))
       with momentum = (momentum (net neuron))
       with output = (output neuron)
       for cx in (cxs neuron)
       for delta = (* learning-rate (err (target cx)) output)
       do (incf (weight cx) (+ delta (* momentum (delta cx))))))
  (:method ((layer t-layer))
    (loop for neuron across (neuron-array layer)
       do (update-weights neuron)))
  (:method ((net t-net))
    (loop for layer in (layers net) 
       do (update-weights layer))))

(defmethod output-layer-error ((net t-net) (outputs vector))
  (loop for neuron across (neuron-array (output-layer net))
     for expected-output across outputs
     for actual-output = (output neuron)
     do (setf (expected-output neuron) expected-output)
     summing
       (* 0.5 (expt (- expected-output actual-output) 2))
     into error
     finally (return (sqrt error))))

(defmethod learn-vector ((inputs vector) (outputs vector) (net t-net))
  (feed net inputs)
  (let ((err (output-layer-error net outputs)))
    (backprop net)
    (update-weights net)
    err))

(defun tset-list-to-tset-vectors (tset)
  "Converts something like
     '((0 0) (1) (0 1) (0) (1 0) (0) (1 1) (1))
   into something like(output neuron
     #((#(0 0) #(1)) (#(0 1) #(0)) (#(1 0) #(0)) (#(1 1) #(1)))"
  (map 'vector 'identity
       (loop for i from 0 below (length tset) by 2 collect
            (list (map 'vector 'identity (elt tset i))
                  (map 'vector 'identity (elt tset (1+ i)))))))

(defgeneric present-vectors
    (t-net t-set iteration start-time report-function time-span-between-reports)
  (:method ((net t-net) (t-set list)
            (iteration integer)
            (start-time integer)
            (report-function function)
            (time-span-between-reports integer))
    (loop
       ;; with vec = (tset-list-to-tset-vectors t-set)
       ;; with shuffled-vector-indices =
       ;;   (shuffle (loop for a from 0 below (length vec) collect a))
       ;; for i in shuffled-vector-indices
       ;; for i from 0 below (length vec)
       ;; for j = 1 then (1+ j)
       for presentation in t-set
       for vector-error = (learn-vector (first presentation)
                                        (second presentation)
                                        net)
       when (and report-function
                 (or (null (last-report-time net))
                     (>= (- (get-internal-real-time) (last-report-time net))
                         time-span-between-reports)))
       do
         (let ((mse (if vector-error-collection
                        (/ (apply '+ vector-error-collection)
                           (float (length vector-error-collection)))
                        1.0)))
           (funcall report-function
                    :net net
                    :elapsed (elapsed start-time)
                    :iteration iteration
                    :mse mse)
           (setf (last-report-time net) (get-internal-real-time)))
       collect vector-error into vector-error-collection
       finally (return (/ (apply '+ vector-error-collection)
                          (float (length vector-error-collection)))))))

(defgeneric randomize-weights (object &key)
  (:method ((neuron t-neuron) &key max min)
    (declare (real max min))
    (if (= max min)
        (loop for cx in (cxs neuron) do
             (setf (weight cx) max)
             (setf (delta cx) 0))
        (loop for cx in (cxs neuron) do
             (setf (weight cx) (+ (random (- max min)) min))
             (setf (delta cx) 0)))
    neuron)
  (:method ((layer t-layer) &key max min)
    (declare (real max min))
    (loop for neuron across (neuron-array layer)
       do (randomize-weights neuron :max max :min min))
    layer)
  (:method ((net t-net) &key (max 1.0) (min (- max)))
    (declare (real max min))
    (loop for layer in (layers net) do
         (randomize-weights layer :max max :min min))
    net))

(defgeneric anneal-weights (object variance)
  (:method ((neuron t-neuron) (variance real))
    (loop for cx in (cxs neuron)
       do (setf (weight cx)
                (+ (weight cx)
                   (/ (* (weight cx) (random variance)) 2)))
       finally (return neuron)))
  (:method ((layer t-layer) (variance real))
    (loop for neuron across (neuron-array layer)
       do (anneal-weights neuron variance)
       finally (return layer)))
  (:method ((net t-net) (variance real))
    (loop for layer in (layers net)
         do (anneal-weights layer variance)
         finally (return net))))


(defun elapsed (start-time)
  (- (get-universal-time) start-time))

(defun write-log-entry (stream message)
  (let* ((log-entry (log-entry message))
         (length-of-message (1- (length log-entry))))
    (write-line log-entry stream :start 0 :end length-of-message)))

(defun default-report-function (&key net elapsed iteration mse)
  (when (> mse (max-mse net)) (setf (max-mse net) mse))
  (when (< mse (min-mse net)) (setf (min-mse net) mse))
  (push (list iteration mse) (mse-list net))
  (with-open-file (stream (log-file net) :direction :output
                          :if-exists :append
                          :if-does-not-exist :create)
    (write-log-entry stream
                     (format nil "~a ~a ~5$ ~4$ ~4$"
                             elapsed iteration
                             mse (min-mse net) (max-mse net)))))

(defun default-status-function (&key net status elapsed iteration mse)
  (when (> mse (max-mse net)) (setf (max-mse net) mse))
  (when (< mse (min-mse net)) (setf (min-mse net) mse))
  (push (list iteration mse) (mse-list net))
  (with-open-file (stream (log-file net) :direction :output
                          :if-exists :append
                          :if-does-not-exist :create)
    (write-log-entry stream
                     (format nil "~a ~a ~a ~5$ ~4$ ~4$"
                             status elapsed iteration
                             mse (min-mse net) (max-mse net)))))

(defun default-logger-function (net message)
  (with-open-file (stream (log-file net) :direction :output
                          :if-exists :append
                          :if-does-not-exist :create)
    (write-log-entry stream message)))

(defun initialize-training (net
                            log-file
                            status-function
                            randomize-weights)
  (setf (log-file net)
        (if log-file
            log-file
            (format nil "/tmp/training-~a.log" (id net))))
  (setf (max-mse net) -1000000.0)
  (setf (min-mse net) 1000000.0)
  (setf (last-anneal-iteration net) 0)
  (setf (anneal net) nil)
  (setf (randomize net) nil)
  (setf (stop-training net) nil)
  (setf (shock net) nil)
  (when status-function
    (funcall status-function
             :net net
             :status "learning"
             :elapsed 0
             :iteration 0
             :mse 1.0))
  (when randomize-weights
    (if (listp randomize-weights)
        (apply #'randomize-weights (cons net randomize-weights))
        (randomize-weights net))
    (setf (mse-list net) nil)))

(defun shock-weights (net
                      target-mse
                      mse
                      rerandomize-weights
                      randomize-weights
                      logger-function
                      annealing
                      i)
  (when (or (and (> mse (* target-mse 10))
                 rerandomize-weights)
            (shock net))
    (if (listp randomize-weights)
        (apply #'randomize-weights (cons net randomize-weights))
        (randomize-weights net))
    (when logger-function
      (funcall logger-function net "randomized weights")))
  (when (and annealing
             (or (shock net)
                 (> (- i (last-anneal-iteration net))
                    annealing)))
    (anneal-weights net 0.1)
    (setf (last-anneal-iteration net) i)
    (when logger-function
      (funcall logger-function "annealed weights")))
  (setf (randomize net) nil)
  (setf (anneal net) nil)
  (setf (shock net) nil))

(defgeneric train (t-net t-set &key)
  (:method ((net t-net)
            (t-set list)
            &key
              (target-mse 0.08)
              (max-iterations 1000000)
              (report-frequency 1000)
              (report-function #'default-report-function)
              (status-function #'default-status-function)
              (logger-function #'default-logger-function)
              (randomize-weights '(:max 1.0 :min -1.0))
              (annealing nil)
              (rerandomize-weights nil)
              (log-file nil))
    (declare (real target-mse)
             (integer max-iterations report-frequency)
             (function report-function status-function))
    (make-thread
     (lambda ()
       (loop
          initially (initialize-training net
                                         log-file
                                         status-function
                                         randomize-weights)
          with start-time = (get-universal-time)
          with time-span-between-reports =
            (truncate (* (/ report-frequency 1000.0)
                         internal-time-units-per-second))
          for i from 1 to max-iterations
          for mse = 1.0 then (present-vectors 
                              net t-set
                              i
                              start-time
                              report-function
                              time-span-between-reports)
          while (and (> mse target-mse) (not (stop-training net)))
          when (or
                (and (> i 1)
                     (> mse (* 10 target-mse))
                     (or rerandomize-weights annealing))
                (randomize net)
                (anneal net)
                (shock net))
          do (shock-weights net target-mse mse rerandomize-weights
                            randomize-weights logger-function annealing i)
          finally (let ((elapsed (elapsed start-time))
                        (status (if (> mse target-mse) "maxi" "target")))
                    (when report-function
                      (funcall report-function
                               :net net
                               :elapsed elapsed
                               :iteration i
                               :mse mse))
                    (when status-function
                      (funcall status-function
                               :net net
                               :status status
                               :elapsed elapsed
                               :iteration i
                               :mse mse))
                    (return (list :elapsed (elapsed start-time)
                                  :iterations i
                                  :error mse
                                  :status status)))))
     :name (format nil "training-~a" (id net))))
  (:documentation "This function uses the standard backprogation of error method to train the neural network on the given sample set within the given constraints. Training is achieved when the target error reaches a level that is equal to or below the given target-mse value, within the given number of iterations. The function returns t if training is achieved and nil otherwise. If training is not achieved, the caller can call again to train for additional iterations. If randomize-weights is set to true or to a value like '(:min -1.0 :max 1.0), which is the default, then the function starts training from scratch. Otherwise, if randomize-weights is set to nil, training resumes from where it left of in the last call.  This function accepts callback parameters that allow the function to periodically report on the progress of training. t-net is a list of training vectors. Each training vector consists of a list that contains an input vector and an output vector.  Here's an example for the exclusive-or problem:
    (list (#(0 0) #(1))
          (#(0 1) #(0))
          (#(1 0) #(0))
          (#(1 1) #(1)))"))

(defgeneric object-freeze (object stream)
  (:method ((net t-net) (s stream))
    (format s "~a~%~a ~a~%"
            (topology net) (learning-rate net) (momentum net))
    (loop for layer in (layers net) do (object-freeze layer s)))
  (:method ((layer t-layer) (s stream))
    (loop for neuron across (neuron-array layer) do (object-freeze neuron s)))
  (:method ((neuron t-neuron) (s stream))
    (format s "~{~{~a ~a~}~^ ~}~%"
            (loop for cx in (cxs neuron)
               collect (list (weight cx) (delta cx))))))

(defun ann-freeze (net)
  (with-output-to-string (s) (object-freeze net s)))

(defgeneric object-thaw (object stream)
  (:method ((net t-net) (s stream))
    (loop for layer in (layers net) do (object-thaw layer s))
    (feed net (make-array (neuron-count (input-layer net)) :initial-element 0))
    net)
  (:method ((layer t-layer) (s stream))
    (loop for neuron across (neuron-array layer) do (object-thaw neuron s)))
  (:method ((neuron t-neuron) (s stream))
    (loop for cx in (cxs neuron) do
         (setf (weight cx) (read s))
         (setf (delta cx) (read s)))))

(defun ann-thaw (string)
    (with-input-from-string (stream string)
      (object-thaw (make-instance 't-net
                                  :topology (read stream)
                                  :learning-rate (read stream)
                                  :momentum (read stream))
                   stream)))

(defgeneric layout-neurons (t-net &key)
  (:method ((net t-net) &key
            (width 200.0)
            (height 100.0)
            (h-margin 20.0)
            (v-margin 20.0))
    (declare (real width height h-margin v-margin))
    (let* ((c-width (- width (* h-margin 2)))
           (v-delta (/ (- height (* v-margin 2))
                              (1- (length (layers net))))))
      (loop
         for layer in (layers net)
         for y = v-margin then (+ y v-delta)
         for h-delta = (/ c-width (neuron-count layer))
         do (layout-neurons layer :y y :delta h-delta :margin h-margin)))
    net)
  (:method ((layer t-layer) &key y delta margin)
    (declare (real y delta margin))
    (loop
       for neuron across (neuron-array layer)
       for x = (+ margin (/ delta 2)) then (+ x delta)
       do
         (setf (x-coor neuron) x)
         (setf (y-coor neuron) y))
    layer))

(defun read-data (net csv-file &key outputs-first limit output-labels)
  (let (tset)
    (with-lines-in-file (row csv-file)
      (when (not (zerop (length (trim row))))
        (let* ((values (split-n-trim row :on-regex ","))
               (numbers (cond ((and outputs-first output-labels)
                               (cons (car values)
                                     (mapcar #'parse-number (cdr values))))
                              (output-labels
                               (append (mapcar #'parse-number (butlast values))
                                       (last values)))
                              (t (mapcar #'parse-number values))))
               inputs
               outputs)
          (if outputs-first
              (setf outputs (if output-labels
                                (car numbers)
                                (subseq numbers 0 (length (outputs net))))
                    inputs (if output-labels
                               (cdr numbers)
                               (subseq numbers (length (outputs net)))))
              (setf inputs (if output-labels 
                               (butlast numbers)
                               (subseq numbers 0 (car (topology net))))
                    outputs (if output-labels
                                (car (last numbers))
                                (subseq numbers (car (topology net))))))
          (when output-labels
            (setf outputs (loop with position = (position outputs output-labels
                                                          :test 'equal)
                             for a from 0 below (length (outputs net))
                             collect (if (= a position) 1.0 0.0))))
          (let ((input-vector (apply #'vector inputs))
                (output-vector (apply #'vector outputs)))
            (push (list input-vector output-vector) tset))))
        (when limit
          (decf limit)
          (when (zerop limit) (return tset))))
    tset))

(defun evaluate-training (net test-set)
  (let ((total 0.0)
        (correct 0.0))
    (loop for (inputs outputs) in test-set
       while inputs do
         (incf total)
         (let ((targets (feed net inputs)))
           (when (loop for output across outputs
                    for target across targets
                    always (or (and (>= target 0.5) (>= output 0.5))
                               (and (< target 0.5) (< output 0.5))))
             (incf correct)))
       finally (return (/ (truncate (* (/ correct total) 10000)) 100.0)))))

(defun circle-training (count &key stats)
  (if stats
      (loop for a from 1 to count
         for x = (* (if (zerop (random 2)) 1 0) (random 1.0))
         for y = (* (if (zerop (random 2)) 1 0) (random 1.0))
         for c = (sqrt (+ (* x x) (* y y)))
         maximizing x into maxx
         maximizing y into maxy
         minimizing x into minx
         minimizing y into miny
         summing c into sumc
         finally (return (list :max-x maxx :max-y maxy
                               :min-x minx :min-y miny
                               :average-diameter (/ sumc count))))
      (loop with diameter = 0.429
         for a from 1 to count
         for x = (* (if (zerop (random 2)) 1 0) (random 1.0))
         for y = (* (if (zerop (random 2)) 1 0) (random 1.0))
         for c = (sqrt (+ (* x x) (* y y)))
         collect (list (vector x y) (vector (if (> c diameter) 0 1))))))
     
(defun circle-training-1hs (count)
  "First output represents 'yes', second output 'no'"
  (loop with diameter = 0.429
     for a from 1 to count
     for x = (* (if (zerop (random 2)) 1 0) (random 1.0))
     for y = (* (if (zerop (random 2)) 1 0) (random 1.0))
     for c = (sqrt (+ (* x x) (* y y)))
     collect (list (vector x y)
                   (if (> c diameter)
                       (vector 0.0 1.0)
                       (vector 1.0 0.0)))))

(defun xor-training ()
  (list (list #(0.0 0.0) #(1.0))
        (list #(0.0 1.0) #(0.0))
        (list #(1.0 0.0) #(0.0))
        (list #(1.0 1.0) #(1.0))))

(defun evaluate-label-training (net presentation)
  (let* ((max (apply #'max (map 'list 'identity (second presentation))))
         (target (position max (second presentation)))
         (output (feed net (first presentation)))
         (rmax (apply #'max (map 'list 'identity output)))
         (result (position rmax output)))
    (list :target target :result result)))


(defun xor-training-1hs ()
  "First output represents 'yes', second output 'no'"
  (list (list #(0.0 0.0) #(1.0 0.0))
        (list #(0.0 1.0) #(0.0 1.0))
        (list #(1.0 0.0) #(0.0 1.0))
        (list #(1.0 1.0) #(1.0 0.0))))

(defun generate-counting-data (n)
  (loop for a from 1 to n
     for digit = (random 10)
     for input-lit = (loop with places = nil 
                  for b from 1 to digit
                  for place = (loop for x = (random 10)
                                 while (member x places)
                                 finally (return x))
                  do (push place places)
                  collect place)
     for output = (loop for a from 0 below 10 collect (if (= a digit) 1 0))
     for input = (loop for c from 0 to 9 collect (if (member c input-lit) 1 0))
     collect (list (map 'vector 'identity input)
                   (map 'vector 'identity output))))
