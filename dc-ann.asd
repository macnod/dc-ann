;;;; dc-ann.asd

(asdf:defsystem #:dc-ann
  :description "Simple implementation of a multilayer backprop neural network."
  :author "Donnie Cameron <macnod@gmail.com>"
  :license "MIT License"
  :depends-on (:dc-utilities)
  :serial t
  :components ((:file "package")
               (:file "dc-ann")))

