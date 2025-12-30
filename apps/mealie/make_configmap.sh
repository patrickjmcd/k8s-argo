envsubst < .env.template > .env
kubectl delete configmap mealie-config --ignore-not-found
kubectl create configmap mealie-config --from-file=.env
rm .env