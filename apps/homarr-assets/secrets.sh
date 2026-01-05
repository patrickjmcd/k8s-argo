kubectl delete secret db-encryption -n homarr --ignore-not-found
kubectl create secret generic db-encryption \
--from-literal=db-encryption-key="$HOMARR_ENCRYPTION_KEY" \
--namespace homarr

kubectl delete secret db-secret -n homarr --ignore-not-found
kubectl create secret generic db-secret \
--from-literal=db-url="postgresql://homarr:$HOMARR_DB_PASSWORD@$POSTGRES_ADDRESS:5432/homarrdb" \
--namespace homarr