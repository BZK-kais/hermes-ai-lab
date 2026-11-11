# PAWS-Wiki · comparaison des adaptateurs LoRA après 512 itérations

Campagne du 9 novembre 2026, vérification finale le 10 novembre · Hermes AI Lab 0.7 · MLX Metal sur Mac M5

**Aucune configuration qualifiée et aucun gain validé.** Sur les 192 nouvelles paires, l'ancien LoRA obtient **159/192** et le nouveau **158/192**. Le nouveau retrouve davantage de doublons et classe à tort 33 paires différentes comme équivalentes, contre 7 pour l'ancien. Aucune activation. Les 192 paires réservées sont désormais consommées. [Audit de la campagne](../../.local/lora-improve-v1/evidence.json)

## Protocole et entraînement

Le diagnostic initial trouvait 21/32 bonnes réponses pour l'ancien LoRA, dont 12/16 sur les paires différentes. Il lui manquait trois réponses globalement et deux sur cette classe pour franchir les seuils. L'expérience a testé une exposition plus longue sur les mêmes paires et une augmentation de la diversité à nombre de présentations égal. L'objectif du contrôleur est passé de `latency` à `quality`. Les seuils numériques et les budgets ont été conservés, dont 75 % global, 85 % sur `different`, 50 % sur `same`, 2 000 ms en p95, 4 Gio pour l'indicateur mémoire MLX et un gain relatif minimal de 10 %. Les variantes utilisent le même Qwen2.5-1.5B-Instruct 4 bits, le même prompt et la même contrainte de sortie JSON à deux labels. [Diagnostic initial](2026-11-09-lora-dev-diagnosis.md) · [Plan figé](../../.local/lora-improve-v1/plan.json)

| Adaptateur | Paires disponibles | Itérations | Présentations nominales |
| --- | ---: | ---: | ---: |
| Ancien | 1 024 | 256 | 1 024 |
| `repeat` | 1 024 | 512 | 2 048 |
| `diverse` | 4 096, dont 3 072 nouvelles | 512 | 2 048 |

Les nouveaux entraînements ont utilisé des lots de 4, le rang 16 sur 12 couches, un taux de 0,0001 et le masque de perte sur le prompt. La validation MLX est passée de 64 à 32 paires, utilisées pour suivre la perte. Le générateur aléatoire de NumPy est partagé par la validation et l'entraînement. Cette différence peut modifier l'ordre du deuxième passage de `repeat`. `diverse` a parcouru la moitié du fichier de 4 096 paires et les lots exacts n'ont pas été relevés. L'absence de courbes de perte, de checkpoint intermédiaire du même entraînement et d'essais à d'autres graines laisse le surentraînement au rang d'hypothèse. [Données préparées](../../.local/lora-improve-v1/prepared-data-proof.json) · [Rapport repeat](../../.local/hermes-experiment/profiles/ai-lab-teacher/ai-lab/workspace/lora-improve-repeat-01/report.json) · [Rapport diverse](../../.local/hermes-experiment/profiles/ai-lab-teacher/ai-lab/workspace/lora-improve-diverse-01/report.json)

Le développement de 32 paires est identique à celui du contrôleur précédent. Le test final de 192 paires équilibrées vient du split officiel test de PAWS-Wiki. L'audit du corpus vérifie les empreintes et la disjonction des nouvelles familles de test avec les sélections antérieures, l'entraînement élargi et le développement. [Provenance du corpus](../../examples/experiments/lora-improve-v1/provenance.json) · [Audit des données](../../.local/lora-improve-v1/data-proof.json)

## Criblage sur 32 paires

| Adaptateur | Correct | `different` | `same` |
| --- | ---: | ---: | ---: |
| Ancien | 21/32 | 12/16 | 9/16 |
| `repeat` | 20/32 | 4/16 | 16/16 |
| `diverse` | 18/32 | 5/16 | 13/16 |

Les trois variantes échouent aux seuils global et `different`. Parmi les deux nouveaux candidats complets, `repeat` a été retenu pour la comparaison finale selon la règle figée, car son exactitude globale de développement était supérieure à celle de `diverse`. Le criblage n'a évalué que le développement. [Rapport du criblage](../../.local/hermes-experiment/profiles/ai-lab-teacher/ai-lab/workspace/lora-improve-screen-01/report.json) · [Sélection figée](../../.local/lora-improve-v1/selection.json)

## Comparaison finale sur 192 paires

| Indicateur | Ancien LoRA | `repeat` |
| --- | ---: | ---: |
| Correct | 159/192 (82,81 %) | 158/192 (82,29 %) |
| `different` correct | 89/96 (92,71 %) | 63/96 (65,63 %) |
| `same` correct | 70/96 (72,92 %) | 95/96 (98,96 %) |
| Fausses équivalences | 7 | 33 |
| Doublons manqués | 26 | 1 |
| Sorties JSON valides et contraintes vérifiées | 192/192 | 192/192 |
| Pire p95 de génération sur les répétitions | 366,890 ms | 361,087 ms |
| Pire indicateur mémoire MLX, actif et cache échantillonné | 1,415 Go | 1,427 Go |

Les quatre répétitions reproduisent les mêmes verdicts par cas. L'appariement sur 192 cas uniques compte 133 réponses correctes communes, 8 erreurs communes, 25 corrections et 26 régressions. McNemar exact bilatéral donne **p = 1,00**. L'exactitude baisse de 0,52 point. L'ancien adaptateur passe tous les seuils finaux mais échoue en développement. Le nouveau échoue aussi au seuil `different` du test. Le contrôleur conclut à `no_qualified_configuration`. Le test final n'a comparé que l'ancien adaptateur et `repeat`. [Rapport final](../../.local/hermes-experiment/profiles/ai-lab-teacher/ai-lab/workspace/lora-improve-final-01/report.json) · [Audit apparié](../../.local/lora-improve-v1/evidence.json)

Le criblage et la comparaison finale ont abouti via Hermes et MLX, en 152,107 s et 582,903 s respectivement. Les identifiants de jobs sont `99cecab6-c09e-4042-a170-ae5afebb5b19` et `47334b0d-1fbc-47aa-81cf-c61b0d40cef7`. La lecture native du rapport final a réussi depuis le profil propriétaire `teacher` et a été refusée depuis `receiver`. La suite de 215 tests et les deux tests supplémentaires de l'audit ont réussi. [Job de criblage](../../.local/lora-improve-v1/job-screen.json) · [Job final](../../.local/lora-improve-v1/job-final.json) · [Lecture teacher](../../.local/lora-improve-v1/native-read-final-teacher.json) · [Refus receiver](../../.local/lora-improve-v1/native-read-final-receiver.json) · [Suite](../../.local/lora-improve-v1/test-results.txt) · [Tests d'audit](../../.local/lora-improve-v1/evidence-test-results.txt)

La vérification finale confirme la réussite des quatre jobs et les empreintes du modèle, des quatre fichiers de données et des 49 fichiers source figés. [Vérification finale](../../.local/lora-improve-v1/verification.json)

La portée reste celle de paires anglaises publiques PAWS, pas de notes Hermes réelles ni d'une validation en production. Le score global presque inchangé masque ici un déplacement des erreurs : les seuils par catégorie empêchent de qualifier un adaptateur qui manquerait moins de doublons en proposant beaucoup plus de fausses équivalences.
