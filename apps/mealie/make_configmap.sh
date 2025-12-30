envsubst < .env.template > .env
kubectl delete configmap mealie-config --ignore-not-found
kubectl create configmap mealie-config --from-env-file=.env
rm .env