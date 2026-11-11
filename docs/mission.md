# Préparer une campagne locale depuis une mission

`ai_lab_analyze` recense les informations nécessaires et propose des pistes. `ai_lab_mission` assemble ensuite la référence et les changements déclarés dans un protocole contrôlé. Un appel avec `{"request":{"action":"prepare"}}` renvoie les champs bloquants et les questions à résoudre. La préparation vérifie les chemins locaux, le jeu de développement, les empreintes des splits et le budget de génération. Le chargement des modèles et les mesures commencent au lancement explicite.

Placer les données au format [du contrôleur](controller.md#figer-les-données-et-lire-le-verdict) dans une racine autorisée du profil. Le dossier contient `dev.jsonl`, `test.jsonl` et `manifest.sha256`. Le mode complet exige un test réservé frais. Le mode `development` utilise seulement les mesures de développement et accepte un test déjà consommé. Dans les deux modes, la préparation vérifie l’empreinte du test sans décoder ses lignes. Le contrôleur contrôle la disjonction des familles et identifiants au moment du test complet.

La campagne utilise des modèles MLX déjà présents sur un Mac Apple Silicon et `backend_paths.mlx.python` doit désigner un exécutable. Pour une tâche de classification, d’extraction ou de génération, choisir une référence par cas adaptée à `json_exact` ou `exact_text`. `objective.quality_metric` doit indiquer le correcteur choisi, `accuracy`, `exact_match` ou `exact_match_accuracy`. Les plafonds obligatoires portent sur la latence p95 de génération et la mémoire active MLX plus son cache. Le débit décrit les requêtes séquentielles. La concurrence vaut 1, les caches KV sont renouvelés à chaque requête et la dépense externe vaut zéro. Le moteur accepte au plus 2 048 tokens d’entrée après gabarit et 128 tokens de sortie. L’exécution effective valide le gabarit et les actifs du modèle.

## Déclarer la référence et les variantes

`mission.json` ci-dessous illustre les champs et leurs unités. Adapter chaque seuil, limite et chemin aux besoins de la mission. `output_dir` doit être nouveau, sous le workspace du profil et avec un parent existant.

```json
{
  "action": "prepare",
  "mission": {
    "schema_version": 1,
    "task": {"description": "Classer une demande selon son sujet.", "type": "classification", "language": "fr"},
    "objective": {"primary": "quality", "quality_metric": "json_exact", "quality_min": 0.9},
    "workload": {"mode": "interactive", "input_tokens_p95": 120, "output_tokens_p95": 16, "concurrency": 1},
    "constraints": {"data_policy": "local_only", "max_memory_bytes": 4000000000, "max_p95_latency_ms": 1200},
    "budget": {"max_trials": 3, "max_wall_seconds": 600, "max_spend": 0},
    "evidence": {
      "baseline_ref": "classification-reference-v1",
      "evaluator_ref": "classification-json-exact-v1",
      "representative_inputs_ref": "classification-splits-v1",
      "consumed_holdout": false
    }
  },
  "dataset_dir": "/chemin/profil/ai-lab/workspace/classification-splits-v1",
  "output_dir": "/chemin/profil/ai-lab/workspace/classification-campagne-01",
  "baseline": {
    "model": "/chemin/modeles/modele-mlx",
    "system_prompt": "Retourne {\"sujet\":\"facturation\"} ou {\"sujet\":\"support\"} selon la demande.",
    "input_encoding": "plain",
    "max_tokens": 24,
    "prefill_step_size": 256
  },
  "grader": "json_exact",
  "repetitions": 2,
  "minimum_relative_gain": 0.05,
  "quality_noninferiority_tolerance": 0.02,
  "max_generated_tokens": 16384,
  "prompt_candidates": ["Détermine si la demande concerne une facture ou une aide technique. Réponds uniquement avec un objet JSON sous la clé sujet, dont la valeur vaut facturation ou support."],
  "prefill_candidates": [2048]
}
```

La référence seule suffit pour une campagne de qualification. Chaque variante reprend sa configuration et remplace un élément déclaré dans `prompt_candidates`, `model_candidates`, `adapter_candidates` ou `prefill_candidates`. Une entrée de `model_candidates` est un objet `{ "model": "/chemin/local" }`. Pour associer un adaptateur compatible au modèle candidat, déclarer `adapter` dans la même entrée. Dans `adapter_candidates`, une entrée `null` retire l’adaptateur de référence. `baseline` peut aussi déclarer `output_constraint` pour les [choix JSON finis](controller.md#réponses-json-contraintes-et-paire-pré-déclarée-06). `minimum_slice_quality` fixe éventuellement des seuils par catégorie. Le plafond de génération réserve les échauffements, toutes les répétitions et, en mode complet, les finalistes. Les doublons et les campagnes excédant les budgets sont refusés avant écriture.

## Lancer et suivre la campagne

```sh
hermes ai-lab mission --args-file mission.json > mission-result.json
jq -e '.start_request' mission-result.json > start.json
hermes ai-lab optimize --args-file start.json
```

Vérifier `status: protocol_ready`, `protocol_path`, `protocol_sha256` et les variantes générées avant de lancer la dernière commande. Le premier résultat conserve aussi les empreintes des splits vérifiés. Au lancement, le contrôleur épingle les sources présentes à cet instant et revérifie leurs empreintes pendant la campagne. L’appel direct à `ai_lab_mission` enveloppe l’objet de `mission.json` sous `request`. L’appel direct à `ai_lab_optimize` enveloppe de la même façon l’objet `start_request`.

Le lancement renvoie un identifiant de job persistant. Suivre ce job avec `ai_lab_job` et lire le verdict via `ai_lab_optimize` avec `action: report` et le chemin de `campaign/report.json`. Le [contrôleur](controller.md#interpréter-les-mesures) décrit la portée des mesures et les critères de qualification.

## Observer un entraînement `mlx.train`

Un entraînement de données MLX `train.jsonl` et `valid.jsonl` se lance séparément avec `ai_lab_run` et l’opération `mlx.train`. Un chemin explicite pour `mlx_lm.lora` est prioritaire. Le profil peut aussi trouver cet exécutable à côté de l’interpréteur `mlx.python`. Les paramètres optionnels `steps_per_report`, `steps_per_eval` et `save_every` définissent les fréquences de perte d’entraînement, de validation et de sauvegarde. Ils acceptent des entiers de 1 à 1 000 000. Sans valeur fournie, le CLI MLX-LM utilise ses propres fréquences.

Consulter `ai_lab_job` avec `action: logs` pendant l’exécution. Les points de contrôle portent un nom tel que `0000100_adapters.safetensors`. Le job vérifie la présence de `adapter_config.json` et de `adapters.safetensors` après la fin du processus. Dans le CLI MLX-LM vérifié localement, la dernière perte de validation affichée précède la dernière mise à jour des poids. L’évaluation de l’adaptateur exporté se fait dans une campagne de comparaison séparée.
