# Système général et premier relevé d'inférence

Vérification du 9 novembre 2026, AI Lab 0.4.0. Les nouveaux outils fonctionnent par le chargeur et le CLI natifs Hermes. Cette passe vérifie l'analyse générale et l'instrumentation ; elle ne qualifie pas une optimisation de modèle.

## Analyse de l'usage et de l'environnement

Trois contrats illustratifs — extraction française, questions sur un corpus évolutif, génération interactive — ont été soumis par `hermes ai-lab analyze`. Les références et seuils de ces exemples sont des hypothèses de démonstration, pas des exigences confirmées de Kais ni des services existants vérifiés.

L'analyse choisit des familles de leviers différentes selon le cas : règles/schéma/prompt pour l'extraction, récupération pour les connaissances changeantes, moteur/cache/modèle pour la génération lente. L'entraînement demande un écart de qualité résiduel déclaré et des exemples annotés. Une demande partielle concernant les agents Hermes du Mac retourne `needs_context` et trois questions, sans inventer l'objectif ni le seuil qualité.

L'inventaire a observé Apple M5, macOS arm64 et 32 Go installés. La mémoire disponible reste inconnue. Les sept entrées de fichiers de modèle sont des artefacts inventoriés, pas sept modèles validés. Runtimes trouvés par les chemins d'inventaire et interpréteurs configurés sont distingués. Aucun endpoint, serveur SSH ou fournisseur n'a été sondé par l'analyse. [Inventaire](../../.local/general-system/environment.json) · [Contrats illustratifs](../../examples/optimization)

## Mesure réelle par le plugin

Job natif `aa3acd27-7e13-408b-9e08-47ba522a39dd`, opération `inference.benchmark`, état `succeeded`. Le lancement a été retardé une fois par l'expiration du contrôle automatique de permission ; la relance autorisée a réussi. Aucun entraînement, téléchargement ou service distant dans cette passe.

Modèle : Qwen2.5-1.5B-Instruct 4 bits déjà présent. MLX 0.32.4, MLX-LM 0.32.2, aucun adaptateur. Empreinte du modèle : `48ff3e71402454dff004929635b3b4f36c1b9477e90d0e2e85749fdac20fee4f`.

Deux prompts de démonstration rédigés pour cette mesure, formatés avec le template chat du modèle, trois répétitions chacun, génération gloutonne de 48 tokens. Un échauffement de huit tokens précède les mesures. Budget demandé et utilisé : 296 tokens, échauffement inclus. Les jeux de qualité PAWS et triage ne sont pas réutilisés.

| Mesure | Prompt court | Prompt plus long |
| --- | ---: | ---: |
| Tokens d'entrée après template | 46 | 424 |
| Générations mesurées | 3 | 3 |
| Tokens de sortie par génération | 48 | 48 |
| Médiane du délai au premier événement token | 46,26 ms | 126,03 ms |
| Médiane du décodage entre premier et dernier tokens | 125,50 tokens/s | 121,65 tokens/s |

Sur les six appels réunis : médiane TTFT 86,16 ms, p95 descriptif 127,33 ms ; médiane de durée 474,62 ms ; médiane de décodage 122,91 tokens/s. Le chargement après calcul d'empreinte prend 266,64 ms. Mélanger les deux longueurs n'en fait pas une distribution de trafic représentative.

Pic actif MLX : 1 428 420 852 octets. Maximum du cache échantillonné : 29 003 838 octets. Maximum RSS du processus : 1 162 952 704 octets. Ces compteurs ont des périmètres et moments différents, ne sont pas additifs et ne donnent pas la mémoire libre du Mac. Le pic actif fourni par MLX peut dépasser les maxima échantillonnés aux événements de génération.

## Portée exacte

Le modèle est chargé une fois dans un processus neuf ; l'allocateur est échauffé, chaque génération crée un nouveau cache KV et les fichiers ont déjà été lus pour leur empreinte. Il ne s'agit donc pas d'un test de cache disque froid, ni d'un test de réutilisation de préfixe. Le chargement, la tokenisation, le réseau et les outils d'un agent sont exclus de la durée de génération mesurée. Le premier token peut être un token sans texte visible, y compris EOS sur d'autres requêtes.

La concurrence vaut 1. Les quantiles sont descriptifs sur six appels, sans engagement de SLA. La qualité des réponses n'est pas notée ; aucune comparaison avant/après n'a été exécutée. Les champs `quality_evaluated` et `optimization_gain_verified` restent explicitement à `false`. La mesure vérifie une capacité de profilage réutilisable, pas la supériorité de cette configuration.

## Preuves

- [Rapport complet et traces](../../.local/hermes-experiment/profiles/ai-lab-teacher/ai-lab/workspace/inference-profile-01/report.json).
- [Requête locale de démonstration](../../.local/general-system/benchmark-request.json).
- [Analyse de la demande partielle](../../.local/general-system/mac-partial-analysis.json).
- [Registre Hermes réel](../../.local/general-system/hermes-registration.txt) : 18 outils, huit dispatchs ; [158 tests réussis](../../.local/general-system/test-results.txt), dont transports locaux et contrôles des nouveaux outils.

Les prochaines mesures doivent utiliser des entrées représentatives et joindre qualité et ressources au même candidat, conformément au [plan général](../PRODUCT-PLAN.md).
