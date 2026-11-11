# Hermes AI Lab 1.1

AI Lab est un plugin natif pour préparer une mission, comparer des configurations de modèle et utiliser celle qui satisfait les critères fixés. Il expose **22 outils Hermes** et le skill **`ai-lab:workflow`**. Les jobs, les rapports et les configurations actives sont rattachés au profil. Le même contrat est accessible par le CLI `ailab`, MCP stdio et une API HTTP locale.

## Installer dans Hermes

Décompresser l’archive du plugin, puis utiliser **le Python de l’installation Hermes** (Hermes 0.23.1 ou plus récent) avec un profil existant.

```sh
/chemin/vers/python-hermes /chemin/archive/ai-lab/scripts/install_local.py \
  --home /chemin/profil-hermes
```

L’installateur copie le plugin dans `plugins/ai-lab` et l’active pour la prochaine session. Il préserve toute installation différente déjà présente. L’archive contient le code, la documentation et les exemples. Le wheel fournit le code et un jeu de vérification. Les deux distributions excluent les poids locaux, les secrets et l’état utilisateur.

Pour mettre à jour une copie installée depuis une release, relancer `install_local.py --home /chemin/profil-hermes --upgrade` depuis la nouvelle archive. L’installateur vérifie la copie existante et retourne `backup_path` lorsqu’il crée une sauvegarde.

Dans le profil choisi, déclarer les répertoires locaux existants et l’interpréteur de l’environnement MLX. Chaque chemin est ajouté par une option explicite de l’opérateur. `setup --check` consulte l’état du profil et la présence de l’exécutable.

```sh
hermes ai-lab setup \
  --model-root /chemin/modeles \
  --data-root /chemin/donnees \
  --mlx-python /chemin/venv-mlx/bin/python
hermes ai-lab setup --check
hermes ai-lab profile
```

Ouvrir une nouvelle session Hermes. Exemple de demande : « Utilise `ai-lab:workflow` pour rendre mon modèle local plus rapide sur ce jeu d'extraction JSON. Choisis les réglages compatibles, compare-les et conserve les résultats pour les prochains essais. » Le skill prend en charge les choix techniques ; il demande seulement le contexte métier ou les accès réellement manquants.

## Accès et limites de la version 1.1

Les accès facultatifs se configurent dans les paramètres du plugin Hermes. Les clés déclarées sont lues par `agent.secret_scope.get_secret` dans le profil actif. Les connexions contiennent uniquement une référence telle que `hermes:AI_LAB_HF_TOKEN`. Les anciennes références `env:`, `file:`, `gcp:` et `native` sont refusées. Le plugin ne lit aucun coffre de credentials, ne lance pas `gcloud` et ne réutilise pas les identifiants implicites des SDK. Voir [la migration des accès](docs/credential-migration.md) et [les vérifications 1.1](docs/releases/2026-11-11-credentials.md).

Google utilise le client OAuth de l’utilisateur déclaré dans Hermes et `hermes auth add ai-lab:google`. Hermes conserve et renouvelle les jetons. GCS expose uniquement les métadonnées en lecture ; les transferts GCS et l’exécution SSH attendent une route Hermes adaptée et retournent une erreur explicite. Les transferts HF/S3 s’exécutent dans la session active avec un reçu de job, sans reprise automatique après perte de session. Les jobs locaux restent détachés.

**Comportements à connaître avant installation :** les opérations explicites peuvent appeler Hugging Face, Runpod, AWS S3, Google Storage et les endpoints configurés ; lire les modèles et données sous les répertoires autorisés ; écrire les artefacts et rapports du profil ; lancer des exécutables locaux et des jobs longs. Aucun compte cloud, téléchargement de poids, entraînement ou coût distant n’est engagé par la découverte. Les budgets déclarés ne plafonnent pas la facturation fournisseur. Les secrets des appels SDK passent uniquement en mémoire dans un canal stdin privé ; cela ne constitue pas une isolation contre du code exécuté sous le même compte système.

## Boucle automatique dans le checkout de développement

`ai_lab_mission` propose `autoplan`, `autostart`, `autoreport` et `lessons`. À partir de la tâche, de l'objectif, du modèle et du jeu local, le planificateur choisit des essais compatibles, explicite ses valeurs par défaut et réutilise les résultats correspondant au même contexte. `autostart` lance la campagne bornée ; `autoreport` vérifie le job local avant de retenir les leçons. L'exploration reste en développement lorsqu'il manque un contrat de qualification. Les seuils fournis sont conservés.

L'agent s'appuie sur des recettes conditionnelles issues de projets étudiés. Les options CUDA, MoE et distribuées encore non exécutables sont distinguées des essais MLX disponibles. Cette boucle n'entraîne pas les poids de l'agent et n'active pas un candidat. Voir [la boucle et ses preuves](docs/recipes/learning-loop.md). Les archives 1.0 déjà produites restent des instantanés antérieurs à cette évolution locale.

## De la mission à l’inférence

1. **Préparer.** `ai_lab_analyze` recense le contexte manquant. `ai_lab_mission` construit un protocole à partir du jeu de données, de la référence, des variantes déclarées, des seuils et du budget. Le test réservé reste opaque pendant la préparation. Voir le [guide de mission](docs/mission.md).
2. **Mesurer.** Transmettre `start_request` à `ai_lab_optimize`, suivre le job avec `ai_lab_job`, puis lire le rapport. Le contrôleur sélectionne sur le développement et évalue la référence et le finaliste sur le test réservé. Une campagne limitée au développement sert à la présélection. Voir les [règles de décision](docs/controller.md).
3. **Capturer et activer.** Avec `ai_lab_registry`, `capture` enregistre la configuration acceptée à partir d’un job complet du profil. `activate` vérifie les actifs et désigne l’entrée à utiliser. La réussite technique du job et le verdict de qualification sont deux informations distinctes.
4. **Générer.** `generate` soumet un job avec la configuration active, figée au moment de la soumission. `result` récupère son statut ou la réponse vérifiée.
5. **Transférer.** `export` produit un ZIP avec les réglages d’inférence, les empreintes, le résumé d’évaluation et l’adaptateur éventuel. `import` l’associe aux poids de base locaux correspondants et crée une entrée inactive. Pour l’utiliser, qualifier sa configuration dans le profil cible, capturer le résultat puis activer cette nouvelle entrée.
6. **Revenir en arrière.** `rollback` restaure l’activation précédente, y compris l’état initial sans configuration active. Voir les [commandes du registre](docs/registry.md).

Les commandes natives `hermes ai-lab mission`, `optimize`, `job` et `registry` acceptent `--args-file`. La [démonstration reproductible](examples/lifecycle-v1/README.md) prépare 16 cas synthétiques où la valeur à recopier figure dans un champ JSON. Elle vérifie le cycle de qualification sur cette tâche et une seule configuration.

Le contrôleur 1.0 utilise **MLX sur Apple Silicon**, une requête à la fois, avec une correction par correspondance exacte en JSON ou en texte. Les résultats portent sur les données, le modèle, les réglages et le matériel mesurés. Dans l’expérience PAWS historique, l’ancien LoRA obtient **159/192** et le nouveau **158/192**, avec **7 et 33 fausses équivalences** respectivement. La campagne reste non qualifiée. [Rapport PAWS](docs/experiments/2026-11-09-lora-improve.md)

## Opérations et suivi des entraînements

`hermes ai-lab operations` ou `ailab operations` expose les **21 opérations nommées**, leurs paramètres et leurs prérequis. Elles couvrent Hugging Face, MLX, llama.cpp, TRL/PEFT, Unsloth, LLaMA-Factory, NeMo, lm-eval, l’export de sessions Hermes et les transferts HF/S3. L’accès à une opération dépend du moteur, de ses dépendances et des connexions configurées. L’exécution sur NVIDIA et chez les fournisseurs de calcul distants reste à vérifier sur les environnements cibles. [Contrats des connecteurs](docs/provider-adapters.md)

Pour `mlx.train`, fournir le modèle, le dossier de données et un nouveau dossier de sortie sous le workspace. `steps_per_report`, `steps_per_eval` et `save_every` règlent la fréquence des pertes d’entraînement, des pertes de validation et des checkpoints. Le CLI MLX-LM utilise ses propres fréquences si ces paramètres sont omis.

```sh
ailab plan mlx.train --params-file /chemin/entrainement.json
ailab run mlx.train --params-file /chemin/entrainement.json
ailab jobs logs IDENTIFIANT
```

Les points de contrôle sont numérotés. Le job vérifie la présence des fichiers finaux `adapter_config.json` et `adapters.safetensors`. Pour qualifier l’adaptateur exporté, l’ajouter explicitement à une campagne d’évaluation. [Suivre un entraînement MLX](docs/mission.md#observer-un-entraînement-mlxtrain)

## CLI indépendant, MCP et HTTP

Installer le wheel avec Python 3.9 ou plus récent, dans un environnement dédié. Le stockage reste extérieur au paquet. Avec un profil Hermes configuré il se trouve sous `ai-lab/` dans ce profil, et en mode standalone neuf sous le répertoire de données utilisateur `hermes-ai-lab`.

```sh
python3 -m pip install /chemin/hermes_ai_lab-1.1.0-py3-none-any.whl
ailab setup --check
ailab operations
```

L’extra `mcp` fournit le SDK pour `ailab mcp`. L’extra `providers` fournit les SDK des connecteurs correspondants. `ailab api --port 8765` ouvre une API d’outils sur `127.0.0.1`, protégée par `AI_LAB_HTTP_API_KEY` configurée dans Hermes (au moins 32 caractères). Aucun fichier de jeton n’est créé ni lu. Sans le runtime Hermes et un accès déclaré, les opérations authentifiées échouent explicitement. `GET /health` donne l’état du service. `GET /tools` et `POST /tools/call` exigent le jeton Bearer. Un appel contient `name` et `arguments`, par exemple `{"name":"ai_lab_operations","arguments":{}}`. Cette interface sert au dispatch des outils AI Lab. Les clients OpenAI Chat Completions utilisent un autre contrat.

Pour les résultats datés, les artefacts de livraison et les limites de vérification, consulter le [rapport 1.0](docs/releases/2026-11-11-v1.md), [STATUS.md](STATUS.md) et le [contrat de livraison](docs/RELEASE-CONTRACT.md).
