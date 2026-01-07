envsubst < .env.template > .env
kubectl delete configmap -n pocketid pocketid-config --ignore-not-found
kubectl create configmap -n pocketid pocketid-config --from-env-file=.env
rm .env