envsubst < up.template.conf > up.conf
kubectl delete configmap unpoller-config --ignore-not-found
kubectl create configmap unpoller-config --from-file=up.conf
rm up.conf