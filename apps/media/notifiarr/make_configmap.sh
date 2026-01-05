envsubst < notifiarr.template.conf > notifiarr.conf
kubectl delete configmap notifiarr-config -n media --ignore-not-found
kubectl create configmap notifiarr-config --from-file=notifiarr.conf -n media
rm notifiarr.conf