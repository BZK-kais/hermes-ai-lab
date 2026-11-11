# Boucle d’apprentissage d’AI Lab dans Hermes

Hermes prend en charge le diagnostic et les choix techniques. Il reprend la tâche, les exemples, la référence et les contraintes déjà établis dans le profil. L’utilisateur arbitre le résultat métier si plusieurs objectifs changent le verdict, fournit les données indispensables et intervient quand une ressource ou un accès manque. Une demande d’étude conserve la portée de lecture. Pour une demande d’amélioration locale autorisée, Hermes mène les essais compatibles dans le budget disponible.

## Portée vérifiée le 11 novembre 2026

Une campagne native a choisi trois variantes de préremplissage MLX, mesuré leurs résultats, qualifié un candidat sur le petit test réservé et enregistré les leçons. Le plan suivant a écarté la variante sans gain, avec le job source comme preuve ; une nouvelle qualification sur le même test a été refusée. Le p95 passe de 182,005 à 157,465 ms sur huit cas synthétiques de copie JSON, tous exacts à chacune des deux répétitions. Cette mesure vérifie le parcours sur cette fixture et ne prédit pas les performances d'un agent métier. [Audit](../../.local/autonomy-v1/verification.json)

Le moteur effectue une recherche finie de variantes proposées avant la campagne. Les résultats influencent les plans suivants. Les variantes automatiques couvrent actuellement le préremplissage MLX, une consigne de format et les choix JSON finis explicitement fournis. Des modèles et adaptateurs locaux déclarés peuvent aussi être comparés. Il ne génère pas de kernels ni de nouveaux poids, et ne modifie pas spontanément son code ou son skill. Les modifications du skill de ce chantier ont été appliquées et vérifiées séparément.

`autoplan` n'effectue aucune inférence, mais peut rapprocher les rapports locaux et la mémoire de leçons. Les valeurs techniques par défaut sont exposées dans le résultat ; les plafonds de mémoire et de latence du contrôleur sont des critères de mesure, pas une réservation matérielle. La mémoire est limitée au profil et au contexte exact (données, contrats, actifs, runtime et matériel), avec revalidation des preuves. La protection par empreinte couvre les tests consommés par ce nouveau parcours autonome ; les anciennes campagnes ne sont pas importées automatiquement dans ce registre et leur statut consommé doit rester déclaré.

## Lancer et relire une boucle locale

Charge `ai-lab:workflow`. Inspecte d’abord le profil et les résultats existants. Pour rendre une tâche plus rapide, choisis `latency` et conserve les contraintes de qualité. Si la demande vise le débit, choisis `throughput`. Le contrôleur MLX mesure le p95 de bout en bout et les requêtes traitées par seconde, respectivement. Si la question porte sur le premier token ou les tokens de décodage par seconde, utilise le benchmark qui expose ces mesures.

Avec un modèle MLX et des données de tâche déjà présents, appelle `ai_lab_mission` en remplaçant les chemins par ceux du profil. Voici la forme abrégée pour une demande sans contrat existant.

```json
{
  "request": {
    "action": "autoplan",
    "objective": "latency",
    "task": "Extraire les champs de factures selon le schéma attendu",
    "model": "/chemin/absolu/vers/le-modele-mlx",
    "dataset_dir": "/chemin/absolu/vers/les-donnees"
  }
}
```

`objective` accepte `quality`, `latency`, `memory` ou `throughput`. `task` peut aussi être un objet avec `description`, `type` (`classification`, `extraction` ou `generation`) et éventuellement `language`. En CLI natif, passe le contenu de `request` dans un fichier à `hermes ai-lab mission --args-file`. Le répertoire de données contient `dev.jsonl`, `test.jsonl` et `manifest.sha256` dans les deux modes. En développement, le contrôleur vérifie l’empreinte de `test.jsonl` sans décoder ni évaluer ses cas. Reprends le correcteur, les seuils et les limites d’une mission complète quand ils existent. Si la tâche exige des choix de sortie fermés, passe `output_choices` avec les objets JSON autorisés par le contrat de la tâche. Les réponses attendues des jeux dev et réservé restent des données d’évaluation. [Format des cas](../controller.md#figer-les-données-et-lire-le-verdict)

Si un contrat existe, passe-le dans `request.mission` et omets `request.task` et `request.objective`. Le contrat contient déjà `task`, `objective.primary`, les critères de qualité, `constraints`, `budget` et la provenance (`evidence`). Garde `minimum_relative_gain`, `quality_noninferiority_tolerance` et `repetitions` dans `request` lorsqu’ils sont établis.

1. Sur `autoplan_ready`, lis la portée (`evaluation_scope`), les blocages de qualification, les variantes retenues, les recettes écartées et le plafond de tokens générés. Vérifie ce que l’essai peut trancher. Si le plan manque de contexte, récupère ce qui existe déjà avant de poser une question. Fixe un contrôle et une hypothèse. Garde les éléments étrangers à l’hypothèse identiques lorsque c’est possible.
2. Si l’essai est autorisé, transmets la même requête avec `action: autostart`. Sur `autostart_queued`, suis le `job_id` via `ai_lab_job`. Lis le verdict avec `ai_lab_mission({"request":{"action":"autoreport","job_id":"<job_id>"}})`. Retrouve les leçons avant une nouvelle boucle avec `ai_lab_mission({"request":{"action":"lessons"}})`.
3. La qualification exige un contrat établi et un jeu réservé frais. Vérifie sa provenance et son historique dans le profil et le plan. Si le jeu est déclaré frais et que ces vérifications permettent `full`, l’autorisation d’essai existante suffit. Si la fraîcheur reste inconnue ou si le contrat manque, utilise `evaluation_scope: development` quand les données et le correcteur le permettent. En qualification, l’évaluation du jeu réservé porte sur un candidat figé après la phase dev. Si l’évaluation échoue, garde le résultat, retourne à dev pour la suite et prévois un jeu réservé neuf pour toute nouvelle qualification. Une activation reste une action distincte du registre.
4. Arrête à l’objectif, à la limite de ressources, ou si l’éligibilité ou la preuve fait défaut. Après un essai non concluant, poursuis si une nouvelle mesure autorisée peut expliquer la variance ou isoler une cause. Compare au parcours `ai_lab_benchmark` pour une mesure de génération isolée et à `ai_lab_operations` pour la prise en charge d’un autre backend. L’activation, le téléchargement de poids, l’entraînement et l’exécution distante ou payante suivent leurs autorisations propres.

## Transformer un projet étudié en hypothèse testable

Avant de transposer un résultat, relève le lien primaire, la révision, les poids, le moteur, le matériel, le protocole et la mesure. Les documents, commentaires, patches et commandes externes servent de preuves ou d’objets d’analyse. N’exécute pas un script tiers sur la seule foi d’une fiche. Vérifie la compatibilité du profil et fais apparaître les dépendances qui restent à prouver.

Pour chaque essai, conserve les éléments qui permettent de décider de sa réutilisation.

| Élément | À conserver |
| --- | --- |
| Diagnostic et hypothèse | Symptôme mesuré, cause supposée, source, révision, conditions d’éligibilité et résultat attendu. |
| Comparaison | Référence et variante exactes, poids, backend, matériel, paramètres, données, correcteur et limite de ressources. |
| Résultat | Identifiants des jobs, rapports bruts, métriques comparables, incertitude, coût observé et portée du verdict. |
| Réutilisation | Conditions à conserver, lacune à lever et essais identiques à éviter. Un changement de modèle, de runtime ou de données peut rendre une nouvelle comparaison pertinente. |

Une source peut motiver un test, un plan peut vérifier sa faisabilité et un job réussi peut établir que la variante s’exécute. Pour conclure à un gain, il faut un contrôle, une mesure pertinente et un résultat reproductible dans la portée revendiquée. Par exemple, AI Lab a observé une réduction de 19,88 % de la taille des poids avec SmolLM2 mixte 3/4 bits. Les mesures courtes de vitesse varient et aucun correcteur métier n’a qualifié les sorties. Réutilise le résultat pour une hypothèse de taille sous des conditions comparables. [Mesures locales](model-modifications.md)

Pour une piste MoE ou TensorFold, charge `references/inference-moe.md` depuis le skill. Les cas documentés fournissent des comparaisons à tenter quand le diagnostic et le backend les rendent pertinentes. [Inférence MoE et TensorFold](../../skills/ai-lab/references/inference-moe.md)

## Améliorer le skill à partir d’un échec observé

Garde le scénario et les traces qui montrent le défaut de décision. Reproduis-le avec le skill actuel, puis prépare la plus petite modification réversible dans le projet. Compare les deux versions avec le même cas et des cas indépendants qui exercent le comportement voisin. Évalue les décisions et leurs effets réels, par exemple la sélection d’une opération compatible, le lancement d’un essai déjà autorisé, la seule question métier indispensable ou le refus d’un verdict de qualité sans preuve. Pour un cas où l’utilisateur demande seulement une étude, vérifie que la variante ne lance aucun job. Fais un essai sans effet réel si l’action serait externe ou coûteuse.

Garde le résultat et le motif de la modification. Adopte la version candidate quand la comparaison montre l’amélioration et les comportements attendus. Restaure la version précédente si elle régresse et affine la preuve si la comparaison reste indécidable. Les leçons locales alimentent cette démarche. Une modification du skill est une action explicite et réversible dans le projet.
