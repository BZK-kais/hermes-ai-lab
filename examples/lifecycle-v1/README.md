# Démonstration locale du cycle de qualification

Le modèle recopie la valeur de `requested_label` dans un objet JSON à clé `label`. Les huit cas de développement et les huit cas réservés sont synthétiques. Chaque split contient quatre réponses `same` et quatre `different`, avec des identifiants et des familles distincts. Le manifeste fixe les deux fichiers utilisés par la campagne.

Le score mesure uniquement la copie exacte du champ. Les deux étiquettes sont fournies dans le JSON d’entrée. Une qualification indique que la configuration satisfait les seuils de ce test fonctionnel. La campagne a une seule variante et n’évalue ni l’équivalence de phrases ni l’effet d’un entraînement LoRA.

## Préparer et mesurer

Utiliser un profil Hermes de démonstration configuré pour MLX. Choisir un modèle de conversation MLX déjà présent et autorisé par le profil. Le script accepte aussi `--adapter /chemin/absolu/adaptateur` si l’on souhaite vérifier le chargement d’un adaptateur local compatible dans le même parcours.

```sh
python3 scripts/prepare_demo.py \
  --workspace /chemin/profil/ai-lab/workspace \
  --model /chemin/modeles/modele-mlx > demo.json
hermes ai-lab mission --args-file "$(jq -er '.mission_request_path' demo.json)" > mission-result.json
jq -e '.start_request' mission-result.json > start.json
hermes ai-lab optimize --args-file start.json > search-job.json
```

Le script crée `demo-v1/dataset` et `demo-v1/mission.json` dans le workspace sélectionné. Le premier appel Hermes crée le protocole. Le second lance la campagne avec deux répétitions et des réponses JSON contraintes aux deux étiquettes. Les seuils sont de 100 % de réponses exactes au total et par étiquette, 2 000 ms de latence p95 et 4 Gio de mémoire MLX. Le budget est de 300 secondes et 2 048 tokens générés, échauffements inclus. Un nouvel essai de préparation requiert un workspace où `demo-v1` est disponible.

## Capturer et utiliser la configuration

```sh
jq '{action:"get",job_id:.job.id}' search-job.json > job-status.json
hermes ai-lab job --args-file job-status.json
jq '{action:"report",report_path:.job.metadata.report_path}' search-job.json > report-request.json
hermes ai-lab optimize --args-file report-request.json
```

Quand le job est `succeeded` et que le verdict est `retained_baseline`, reprendre l’identifiant du job pour [capturer la configuration, l’exporter et l’activer dans le profil de démonstration](../../docs/registry.md). Pour tester la configuration active, fournir une nouvelle entrée comme `{"case":"live-check","requested_label":"different"}` et vérifier la réponse `{"label":"different"}`. Le même registre permet de restaurer l’état d’activation précédent avec `rollback`.
