# Comparer des variantes d’inférence avec Hermes

`ai_lab_optimize` compare une liste finie de configurations déclarées à l’avance. Le moteur actuel utilise MLX avec Metal sur Apple Silicon. Le profil Hermes doit disposer de `backend_paths.mlx.python` et de modèles MLX déjà présents sur disque, avec un modèle de conversation pris en charge par le tokenizer. Chaque essai charge sa variante depuis ces fichiers, avec un adaptateur local facultatif. Cette commande fonctionne hors ligne, sans téléchargement ni entraînement, et ne prend pas en charge CUDA.

## Lancer une campagne

Remplacer les chemins absolus d’exemple par ceux du profil actif. Placer le protocole, les données et les modèles sous les racines autorisées de ce profil. Le répertoire de sortie doit être nouveau, sous son workspace, avec un parent déjà existant.

`start.json`

```json
{
  "action": "start",
  "protocol_path": "/chemin/profil/ai-lab/workspace/protocol.json",
  "dataset_dir": "/chemin/profil/ai-lab/workspace/donnees",
  "output_dir": "/chemin/profil/ai-lab/workspace/essais/campagne-01"
}
```

`report-request.json`

```json
{
  "action": "report",
  "report_path": "/chemin/profil/ai-lab/workspace/essais/campagne-01/report.json"
}
```

```sh
hermes ai-lab optimize --args-file /chemin/start.json
hermes ai-lab job --args-file /chemin/job.json
hermes ai-lab optimize --args-file /chemin/report-request.json
```

`start` renvoie un job persistant. Pour suivre son état, écrire dans `job.json` l’objet `{"action":"get","job_id":"IDENTIFIANT_RENVOYÉ"}`, puis lire le rapport à la fin du job. Le CLI prend les objets ci-dessus tels quels. Un appel direct à l’outil Hermes `ai_lab_optimize` enveloppe l’objet dans `{"request": {...}}`.

## Déclarer la comparaison

Exemple de `protocol.json` pour les splits PAWS de [controller-v1](../examples/experiments/controller-v1/), avec des seuils et un modèle à adapter.

```json
{
  "version": "controller-v1",
  "usage": {
    "description": "Classer des paires de phrases anglaises par équivalence de sens.",
    "task_type": "classification"
  },
  "objective": "latency",
  "baseline_id": "reference",
  "grader": {"kind": "json_exact"},
  "constraints": {
    "quality_min": 0.75,
    "minimum_slice_quality": {"same": 0.70, "different": 0.70},
    "max_p95_latency_ms": 2000,
    "max_mlx_memory_bytes": 8000000000
  },
  "budget": {
    "max_wall_seconds": 900,
    "max_trials": 2,
    "max_generated_tokens": 8000
  },
  "repetitions": 2,
  "minimum_relative_gain": 0.05,
  "quality_noninferiority_tolerance": 0.02,
  "variants": [
    {
      "id": "reference",
      "model": "/chemin/modeles/modele-mlx",
      "system_prompt": "Do the two sentences state the same facts? Return only {\"label\":\"same\"} or {\"label\":\"different\"}.",
      "input_encoding": "plain",
      "max_tokens": 16,
      "prefill_step_size": 2048
    },
    {
      "id": "roles",
      "model": "/chemin/modeles/modele-mlx",
      "system_prompt": "Compare who did what to whom, time and quantities. Return only {\"label\":\"same\"} or {\"label\":\"different\"}.",
      "input_encoding": "plain",
      "max_tokens": 16,
      "prefill_step_size": 2048
    }
  ]
}
```

Les objectifs sont `latency`, `quality`, `memory` et `throughput`. Les types de tâche sont `classification`, `extraction` et `generation`. Le correcteur `json_exact` compare les valeurs et les types JSON. `exact_text` compare les textes après suppression des espaces aux extrémités. La qualité est la fraction de réponses exactes selon ce correcteur. Une tâche de génération exige donc des sorties de référence adaptées à cette comparaison stricte.

Le protocole accepte au plus huit variantes, deux à cinq répétitions et 128 tokens générés par requête. `max_trials` borne le nombre de variantes déclarées. Le plafond total de tokens couvre les essais de développement, les finalistes et l’échauffement. Il est contrôlé dès le départ à partir des maxima par requête. Le champ facultatif `adapter` contient un chemin absolu. `input_encoding` vaut `plain` pour une entrée directe ou `json_string` pour une chaîne encodée en JSON. Les pas de préremplissage admis sont 64, 256 et 2048.

## Figer les données et lire le verdict

Le dossier contient `dev.jsonl` et `test.jsonl`, de 1 à 256 lignes chacun, et un `manifest.sha256` au format standard avec exactement une entrée par fichier. Le moteur vérifie les empreintes. Chaque ligne contient exactement `id`, `family`, `slice`, `input` (chaîne) et `expected` (valeur de référence du correcteur). Les identifiants sont uniques par split. Les deux splits doivent être disjoints par identifiant et famille, et contenir chacun les catégories soumises à un seuil. Pour PAWS, `input` est une chaîne qui contient la paire sérialisée en JSON et `expected` est un objet tel que `{"label":"same"}`.

Le moteur évalue les variantes sur `dev`. À chaque répétition, il mélange les cas de façon déterministe et décale l’ordre de passage des variantes. Toutes les variantes d’une même répétition reçoivent les cas dans le même ordre. Les seuils s’appliquent aux valeurs les plus défavorables des répétitions, notamment à la qualité globale et par catégorie. Parmi les variantes admissibles, l’objectif départage en premier, puis la qualité, la latence p95 et l’ordre de déclaration.

Le choix est figé dans `selection.json` avant lecture des cas réservés. Le test compare uniquement la référence et la variante choisie, ou la référence seule si elle arrive en tête. Si aucune variante de développement ne passe les seuils, le test est omis. Aucun second candidat ne remplace un finaliste recalé.

- `qualified_candidate` indique un candidat accepté. Il doit passer les seuils sur le test et rester dans la tolérance de qualité face à la référence à chaque répétition appariée. Si la référence passe les seuils, le gain d’objectif doit aussi être positif et atteindre le minimum demandé. Une référence hors seuil peut donc laisser place à un candidat qualifié sans gain vérifié.
- `retained_baseline` indique que la référence passe les seuils et reste le choix accepté. `no_qualified_configuration` indique qu’aucune configuration n’est acceptée.
- `optimization_gain_verified` distingue le gain mesuré de la qualification. `activation_performed` reste à `false`. Le succès du job indique que le rapport a été produit, indépendamment du verdict de qualification.

## Évaluer seulement le développement (0.7)

Le champ facultatif `evaluation_scope: "development"` limite une campagne à `dev`. Le fichier de test reste requis et son empreinte est vérifiée, mais ses lignes ne sont ni décodées ni mesurées. Le budget prévisionnel couvre seulement le développement. Toutes les mesures doivent terminer correctement avant que le contrôleur fige la sélection et renvoie `development_complete`, avec `test: {}`, aucun candidat accepté et aucune activation. Il peut terminer sans variante admissible : cela reste une mesure, pas une qualification.

Ce mode ne peut pas être combiné avec `holdout_candidate_id`. Pour comparer ensuite un candidat choisi, créer un nouveau protocole complet qui déclare ce candidat et la référence, sans modifier les seuils ni essayer un autre candidat après lecture du test. L'absence de `evaluation_scope` conserve le fonctionnement complet historique ; `"full"` est aussi accepté.

## Réponses JSON contraintes et paire pré-déclarée (0.6)

Une variante peut ajouter une contrainte finie indépendante des réponses attendues :

```json
{
  "output_constraint": {
    "kind": "json_choices",
    "values": [{"label": "same"}, {"label": "different"}]
  }
}
```

Ce fragment complète une variante existante ; les autres champs restent nécessaires. Il impose l'une des valeurs JSON canoniques déclarées, avec le correcteur `json_exact`. Le modèle choisit la valeur à partir de ses logits ; la référence de chaque cas ne lui est jamais transmise. La contrainte est un ensemble fini de 2–32 objets JSON, pas un moteur général de JSON Schema. Les objets sont bornés à 1 024 octets chacun et huit niveaux de conteneurs ; les séquences tokenisées doivent tenir dans le budget de sortie, EOS compris.

Le décodeur masque les jetons hors des séquences autorisées. Après génération, le worker vérifie le chemin de jetons, l'arrêt EOS, le texte exact et les appels effectifs du processeur de logits. Il échoue si la contrainte est ignorée. Chaque génération utilise un nouvel état. Le temps de masquage et les jetons forcés sont inclus dans les mesures ; `decoding_setup_ms` sépare la compilation initiale. L'échauffement contraint peut utiliser le plafond complet de sortie, déjà réservé dans le budget de campagne. `constraint_verified_fraction` expose les vérifications réussies, sans publier les réponses brutes.

Pour une comparaison confirmatoire de deux configurations décidées à l'avance, ajouter au niveau du protocole :

```json
{
  "baseline_id": "qwen-json",
  "holdout_candidate_id": "qwen-lora-json"
}
```

Les deux identifiants doivent désigner des variantes distinctes. `selection_basis` vaut alors `predeclared_pair` ; le candidat apparaît dans le premier rapport, avant toute mesure de développement. La paire est évaluée sur le test réservé même si ses mesures complètes échouent aux seuils de développement. Cela permet de comparer les modèles sans présenter ce passage comme une qualification. Une erreur technique en développement interrompt la campagne avant le test.

Les seuils restent obligatoires sur les deux jeux pour accepter une configuration. Les règles de gain relatif et de qualité par rapport à la référence restent également applicables. Aucun autre candidat n'est essayé sur le test, aucun choix n'est ajusté après ses résultats, et le jeu final devient consommé après l'évaluation. Sans `holdout_candidate_id`, le fonctionnement historique par sélection sur développement reste inchangé.

## Interpréter les mesures

La latence et le débit portent sur les générations séquentielles, hors chargement, tokenisation, échauffement et correction. Chaque génération repart avec un cache KV neuf. Le cache de fichiers du système est réchauffé par le calcul des empreintes et reste non contrôlé. La métrique mémoire combine le pic actif MLX et le maximum observé du cache MLX durant l’essai, chargement et échauffement compris. Le pic RSS du processus est consigné à titre diagnostique.

Les champs `output_parseable_fraction` et `output_schema_matches_fraction` séparent un format illisible d'une structure attendue mais d'une réponse incorrecte. La structure compare les clés, types et longueurs de listes à la référence ; elle ne vérifie pas la valeur sémantique des feuilles. Ces diagnostics ne remplacent pas le score exact. La [vérification sur le M5](experiments/2026-11-09-controller-v1.md) illustre un refus de toutes les variantes.

Les mesures décrivent les variantes déclarées sur ces données et cette machine. Le rapport ne permet d’établir ni un optimum global ni un SLA de bout en bout.
