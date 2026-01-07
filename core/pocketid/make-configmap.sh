envsubst < .env.template > .env
kubectl delete configmap pocketid-config --ignore-not-found
kubectl create configmap pocketid --from-env-file=.env
rm .env