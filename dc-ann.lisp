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
   (training-segments :accessor training-segments :type list
                      :initarg :training-segments :initform nil)
   (last-training-segment :accessor last-training-segment
                          :type string :initform nil)
   (last-tset :accessor last-tset :type list :initform nil)
   (mutexx :accessor mutexx :initform (make-mutex :name "mutexx"))
   (learning-rate :accessor learning-rate :initarg :learning-rate :initform 0.3)
   (momentum :accessor momentum :initarg :momentum :initform 0.8)
   (wi :accessor wi :initform 0)
   (transfer-tag :accessor transfer-tag
                 :initarg :transfer-tag :initform :logistic)
   (layers :accessor layers)
   (next-id :accessor next-id :initform 0)
   (min-mse :accessor min-mse :type real :initform 1000.0)
   (max-mse :accessor max-mse :type real :initform -1000.0)
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
  (unless (log-file net)
    (setf (log-file net) (format nil "/tmp/~a-~a.log" 
                                 (string-downcase (id net))
                                 (unique-name)))
    (with-open-file (f (log-file net) :direction :output :if-exists :supersede)
      nil))
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

(defun bounded (x) x)
  ;; (if (< x 0)
  ;;     (max -1e-10 x)
  ;;     (min 1e10 x)))

(defgeneric fire (object)
  (:method ((neuron t-neuron))
    (transfer neuron)
    (loop for cx in (cxs neuron)
       do (incf (input (target cx)) (* (bounded (weight cx))
                                       (bounded (output neuron)))))
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
    (with-mutex ((mutexx net))
      ;; Copy input values to input layer
      (loop for input across inputs
         for neuron across (neuron-array (input-layer net))
         do (setf (input neuron) input))
      ;; Feed forward
      (fire net)
      ;; Return a vector with the output value of each output-layer neuron
      (map 'vector 'output (neuron-array (output-layer net))))))

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
                    (bounded (output target-neuron)))
                 (funcall (transfer-derivative target-neuron)
                          (bounded (output target-neuron)))))))

(defun backprop-hidden (neuron)
  (loop
     for cx in (cxs neuron)
     for target-neuron = (target cx)
     do (setf (err target-neuron)
              (* (loop for cx in (cxs target-neuron)
                    summing (* (bounded (weight cx)) 
                               (bounded (err (target cx)))))
                 (funcall (transfer-derivative target-neuron)
                          (bounded (output target-neuron)))))))

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

(defgeneric present-vectors (t-net
                             outputs-first
                             output-labels
                             has-header
                             iteration
                             start-time
                             report-function
                             time-span-between-reports)
  (:method ((net t-net)
            outputs-first
            (output-labels list)
            (has-header t)
            (iteration integer)
            (start-time integer)
            (report-function function)
            (time-span-between-reports integer))
    (loop for training-segment in (shuffle (training-segments net))
       for t-set = (read-data net training-segment
                              :outputs-first outputs-first
                              :output-labels output-labels
                              :has-header has-header)
       append (loop
                 for presentation in t-set
                 for vector-error = (learn-vector (first presentation)
                                                  (second presentation)
                                                  net)
                 while (not (stop-training net))
                 when (and report-function
                           (or (null (last-report-time net))
                               (>= (- (get-internal-real-time)
                                      (last-report-time net))
                                   time-span-between-reports))
                           error-collection)
                 do
                   (let ((mse (/ (apply '+ error-collection)
                                 (float (length error-collection)))))
                     (funcall report-function
                              :net net
                              :elapsed (elapsed start-time)
                              :iteration iteration
                              :mse mse)
                     (setf (last-report-time net) (get-internal-real-time)))
                 collect vector-error into error-collection
                 finally (return error-collection))
       into vector-error-collection
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
  (:method ((net t-net) &key (max 0.9) (min 0.1))
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
                     (format nil "~as ~ai ~4$ < ~4$ > ~4$"
                             elapsed iteration
                             (min-mse net) mse (max-mse net)))))

(defun default-status-function (&key net status elapsed iteration mse)
  (when (> mse (max-mse net)) (setf (max-mse net) mse))
  (when (< mse (min-mse net)) (setf (min-mse net) mse))
  (push (list iteration mse) (mse-list net))
  (with-open-file (stream (log-file net) :direction :output
                          :if-exists :append
                          :if-does-not-exist :create)
    (write-log-entry stream
                     (format nil "~a ~as ~ai ~4$ < ~4$ > ~4$"
                             status elapsed iteration
                             (min-mse net) mse (max-mse net)))))

(defun default-logger-function (net message)
  (with-open-file (stream (log-file net) :direction :output
                          :if-exists :append
                          :if-does-not-exist :create)
    (write-log-entry stream message)))

(defun initialize-training (net
                            tset
                            &key
                            (max-segment-size (truncate 1e8))
                            log-file
                            (status-function #'default-status-function)
                            randomize-weights)
  (when log-file
    (setf (log-file net) log-file))
  (setf (training-segments net) (make-training-segments tset max-segment-size)
        (max-mse net) -1000.0
        (min-mse net) 1000.0
        (last-anneal-iteration net) 0
        (anneal net) nil
        (randomize net) nil
        (stop-training net) nil
        (shock net) nil)
  (when status-function
    (funcall status-function
             :net net
             :status "learning"
             :elapsed 0
             :iteration 0
             :mse (min-mse net)))
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
  (funcall logger-function "shock")
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
            (t-set string)
            &key
              (max-segment-size 100000000)
              outputs-first
              output-labels
              has-header
              (target-mse 0.08)
              (max-iterations 1000000)
              (report-frequency 1000)
              (report-function #'default-report-function)
              (status-function #'default-status-function)
              (logger-function #'default-logger-function)
              randomize-weights
              (annealing nil)
              (rerandomize-weights nil)
              (log-file nil)
              (clear-segments t))
    (declare (real target-mse)
             (integer max-iterations report-frequency)
             (function report-function status-function))
    (make-thread
     (lambda ()
       (loop
          initially (initialize-training net
                                         t-set
                                         :max-segment-size max-segment-size
                                         :log-file log-file
                                         :status-function status-function
                                         :randomize-weights randomize-weights)
          with start-time = (get-universal-time)
          with time-span-between-reports =
            (truncate (* (/ report-frequency 1000.0)
                         internal-time-units-per-second))
          for i from 1 to max-iterations
          for mse = (min-mse net) 
          then (present-vectors
                net
                outputs-first
                output-labels
                has-header
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
                    (when clear-segments (clear-segments net))
                    (return (list :elapsed (elapsed start-time)
                                  :iterations i
                                  :error mse
                                  :status status)))))
     :name (format nil "training-~(~a~)" (id net))))
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

(defun read-data (net csv-file &key outputs-first limit output-labels has-header)
  (if (equal (last-training-segment net) csv-file)
      (last-tset net)
      (let (tset past-header)
        (with-lines-in-file (row csv-file)
          (if (or (and has-header past-header)
                  (not has-header))
              (progn
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
              (setf past-header t)))
        (setf (last-training-segment net) csv-file)
        (setf (last-tset net) tset))))

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

(defun evaluate-one-hotshot-training (net test-set &optional show-fails)
  (format nil "~,,2f% accurate"
          (/ (loop for (inputs target-outputs) in test-set
                for target-output-winner = (index-of-max target-outputs)
                for outputs = (map 'list 'identity (feed net inputs))
                for output-winner = (index-of-max outputs)
                when (and show-fails
                          (not (= output-winner target-output-winner)))
                do (format t "~a ~a~%" inputs outputs)
                when (= output-winner target-output-winner)
                summing 1)
             (float (length test-set)))))

(defun evaluate-label-training (net presentation)
  (let* ((max (apply #'max (map 'list 'identity (second presentation))))
         (target (position max (second presentation)))
         (output (feed net (first presentation)))
         (rmax (apply #'max (map 'list 'identity output)))
         (result (position rmax output)))
    (list :target target :result result)))

(defun make-segment-file-names (file
                                file-line-count
                                segment-line-count
                                &optional (target-directory "/tmp"))
  (loop with name = (pathname-name file)
     with extension = (pathname-type file)
     with segment-count = (ceiling (/ (float file-line-count)
                                      segment-line-count))
     for a from 1 to segment-count
     collect (join-paths
              target-directory
              (format nil "~a-~4,'0d.~a" name a extension))))

(defun make-training-segments
    (file &optional (max-segment-size 100000000) (target-directory "/tmp"))
  (let* ((line-count (file-line-count file))
         (average-line-size (truncate (float (lof file)) line-count))
         (segment-line-count (truncate (float max-segment-size) average-line-size))
         (segment-file-names (make-segment-file-names file
                                                      line-count
                                                      segment-line-count
                                                      target-directory)))
    (with-open-file (in file)
      (loop for file in segment-file-names
         do (with-open-file (out file :direction :output :if-exists :supersede)
              (loop with line-count = 1
                 for line = (read-line in nil)
                 when line do
                   (write-line line out)
                   (incf line-count)
                 while (and line (<= line-count segment-line-count))))))
    segment-file-names))

(defun clear-segments (net)
  (loop for segment in (training-segments net)
       do (delete-file segment)))

(defun tset-to-file 
    (net t-set &optional
                 (filename 
                  (unique-file-name 
                   :extension (format nil ".~a.annt" (id net)))))
  (with-open-file (o filename :direction :output :if-exists :supersede)
    (loop for row in t-set do
         (loop for value across (concatenate 'vector (first row) (second row))
            collect value into values
            finally (format o "~{~d~^,~}~%" values))))
  filename)

(defmethod weight-data ((net t-net))
  (let* ((cx-weights (loop for layer in (butlast (dc-ann::layers dc::*xor*))
                        for layer-index = 1 then (1+ layer-index)
                        for cx-index = 0
                        collect 
                          (list
                           (format nil "layer-~a" layer-index)
                           (loop for neuron across (dc-ann::neuron-array layer)
                              appending
                                (loop for cx in (dc-ann::cxs neuron)
                                   do (incf cx-index)
                                   collect 
                                     (list (float cx-index)
                                           (float (dc-ann::weight cx))))))))
         (cx-count (apply #'max (mapcar (lambda (w) (length (second w))) 
                                        cx-weights))))
    (loop for row in (reverse cx-weights)
       for rev = (reverse (second row))
       do (loop while (< (length rev) cx-count)
             do (push (list (float (1+ (length rev))) 0.0) rev))
       collect (list (car row) (reverse rev)))))

(defmethod chart-weight-data ((net t-net))
  (dc-utilities::chart 
    (:line 600 400)
    (loop for row in (weight-data net)
       do (apply #'dc-utilities::add-series row))
    (dc-utilities::set-axis :x "cx")
    (dc-utilities::set-axis 
     :y "w" 
     :label-formatter (lambda (l) 
                        (let ((label (format nil "~,2f" l)))
                          (if (ppcre:scan "\\.[05]0$" label)
                              label
                              ""))))))
