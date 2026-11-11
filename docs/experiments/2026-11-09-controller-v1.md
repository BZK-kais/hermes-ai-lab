# Contrôleur de variantes — vérification native sur le Mac

Le contrôleur `ai_lab_optimize` a exécuté deux campagnes MLX via le plugin Hermes sur le Mac M5 32 Go. **Aucune configuration n'est qualifiée.** Les quatre variantes échouent aux seuils de qualité fixés avant la première génération ; les 64 cas réservés n'ont donc pas été évalués. Le contrôleur n'a activé aucun modèle.

## Capacité vérifiée

Le protocole décrit usage, objectif, référence, variantes, évaluateur, seuils globaux et par tranche, mémoire et budgets. Le worker détecte Apple Silicon et vérifie Metal, fige les empreintes des fichiers, puis sérialise toute la campagne sous le verrou du GPU. Les essais alternent l'ordre des variantes et utilisent le même ordre de cas pour chaque répétition. Les mêmes réponses servent à mesurer qualité et performance.

Les candidats doivent satisfaire les contraintes avant le classement. La sélection est sauvegardée avant le décodage des références du test final ; seuls la référence et le candidat choisi peuvent ensuite être évalués. Un échec final ne déclenche pas l'examen des autres candidats sur ce même test. Ce parcours positif est couvert par des tests simulés ; dans les campagnes réelles présentes, aucune variante ne franchit l'étape de développement.

Il s'agit d'une recherche finie parmi des configurations explicites, sur un matériel observé. La génération automatique de variantes depuis `ai_lab_analyze`, les autres moteurs, la charge concurrente et l'activation ne font pas partie de cette livraison. [Contrat et commandes](../controller.md).

## Protocole fixé avant les essais

- Usage exploratoire : proposer des doublons de mémoire en distinguant les vraies paraphrases des changements de faits. PAWS-Wiki constitue un proxy public anglais ; aucun corpus de notes réelles d'entreprise n'est utilisé.
- 32 nouvelles paires de développement, équilibrées 16/16, puis 64 paires réservées, équilibrées 32/32. Les composantes de phrases communes avec les anciens jeux train/dev/producteur/destinataire sont exclues, ainsi que les recoupements entre les nouveaux splits. Préparation reproductible avec la graine 20261110.
- Révision PAWS `161ece9501cf0a11f3e48bd356eaa82de46d6a09`. Provenance et empreintes : `examples/experiments/controller-v1/provenance.json`. Source humaine du label, paires générées à partir de Wikipédia ; une contamination par le préentraînement reste inconnue. [Source PAWS](https://github.com/google-research-datasets/paws).
- Évaluateur : JSON exact, clé `label` et valeur `same` ou `different`. Ordre des clés et espaces JSON sans effet ; doublons de clés, valeurs non finies, texte supplémentaire et JSON invalide sont rejetés. La classification mesure donc une réponse structurée correcte, pas seulement un choix sémantique latent.
- Objectif : minimiser le p95 de génération parmi les variantes admissibles. Qualité minimale 75 %, reconnaissance des paires différentes 85 %, des équivalentes 50 %, p95 inférieur à 2 000 ms, indicateur mémoire MLX inférieur à 4 Gio. Ces seuils sont des critères d'ingénierie illustratifs, sans validation client.
- Confirmation prévue : qualité au plus deux points sous la référence et gain relatif minimal de 10 % sur l'objectif ; un candidat admissible peut aussi remplacer une référence inadmissible sans annoncer un gain de vitesse. Ces règles comparent des estimations ponctuelles ; elles ne constituent pas un test statistique de non-infériorité.
- Quatre répétitions, au plus 16 tokens par réponse, 32 768 tokens générés échauffements inclus et 1 200 secondes pour la campagne. Aucun entraînement, téléchargement ni achat de calcul.
- Même prompt choisi lors de l'expérience antérieure, même encodage de l'entrée en chaîne JSON. Qwen2.5-1.5B-Instruct 4 bits, SmolLM2-135M 4 bits, puis Qwen avec le LoRA existant, préremplissage de 2 048 ou 64 tokens. MLX 0.32.4, MLX-LM 0.32.2, Python 3.14.6.

Le protocole source a pour SHA-256 `717bba3cf6b44780acaa5a3f1204f49f60713f84df16f72b9e39123a31541144`. Les données dev/test ont respectivement les empreintes `5e53884501f2008cfee1f4094ef68460ddfc2ad8bcf7eb802b2c59767fb9ebfb` et `ff85a612bc684dc128ec4ead0999da88d5e5dbbe229acc0fbf737ea123a8716e`. Les chemins locaux résolus, prompts et empreintes de modèles/adaptateurs sont conservés dans la requête privée de chaque campagne.

## Résultats

Première campagne, job Hermes `c7002d60-3e83-4e61-9dff-5df5e8965641` : **79,367 s**, 512 générations mesurées et 16 échauffements, 4 328 tokens générés au total. Les répétitions ne multiplient pas le nombre de cas indépendants : il reste 32 cas de développement.

| Variante | Réponses JSON correctes | p95 de génération retenu | Indicateur mémoire MLX |
| --- | ---: | ---: | ---: |
| Qwen de base | 1/32 (3,1 %) | 132,7 ms | 1,225 Go |
| Smol de base | 0/32 | 48,0 ms | 0,324 Go |
| Qwen + LoRA | 21/32 (65,6 %) | 181,6 ms | 1,394 Go |
| Qwen + LoRA, préremplissage 64 | 21/32 (65,6 %) | 270,2 ms | 1,113 Go |

Le p95 est calculé par rang supérieur dans chaque répétition, puis le plus élevé est retenu. L'indicateur mémoire additionne le pic actif MLX et le maximum de cache échantillonné ; il ne représente pas la RAM totale du processus, ni une garantie sur un pic de cache non observé. Les Go du tableau utilisent 10⁹ octets. Chargement, tokenisation, échauffement et correction sont exclus du chronométrage des générations. Le RSS est cumulatif et ne sert pas au classement. KV neuf à chaque requête ; cache disque réchauffé par les empreintes et non contrôlé.

Les deux variantes LoRA donnent exactement les mêmes sorties par cas et par répétition. Leur changement de préremplissage est effectif sur toutes les entrées. Les blocs de 64 réduisent l'indicateur mémoire de **20,1 %** mais augmentent le p95 retenu de **48,8 %**. Ce compromis ne constitue pas un gain qualifié : les deux configurations restent sous le seuil de qualité. Sur les 16 cas différents, le LoRA en reconnaît 12 ; sur les 16 équivalents, il en reconnaît 9.

## Diagnostic du format et deuxième vérification

Le faible score de Qwen de base a motivé l'ajout de deux compteurs : sortie JSON décodable et structure compatible avec la référence. Le score, le prompt, les données, les seuils, l'ordre des essais et les paramètres des variantes restent identiques. Les réponses brutes ne sont pas persistées ; leurs empreintes et les indicateurs le sont.

La campagne `fdc360e3-2207-495d-902c-a63e67673976` rejoue ce même protocole sur une nouvelle révision instrumentée : **81,258 s**, mêmes 4 328 tokens, mêmes scores et même refus. Ce rejeu sert à vérifier le diagnostic ; il n'apporte pas de nouveaux cas indépendants.

- Qwen de base : **1/32 sorties JSON décodables**, la seule également correcte. Les 31 autres échouent au format strict ; leur exactitude sémantique n'est pas évaluée séparément.
- Smol : **0/32 sorties JSON décodables**, sorties arrêtées à la limite de 16 tokens.
- Les deux LoRA : **32/32 sorties JSON décodables et de structure attendue**, mais seulement **21/32 réponses correctes**. L'amélioration du format ne résout pas les onze erreurs de sens.

Le rejeu observe des p95 de 144,8 / 109,8 / 178,8 / 341,0 ms dans le même ordre de variantes. La réduction mémoire LoRA reste de 20,1 % ; le surcoût de latence atteint 90,8 %. Cette variation sur des campagnes courtes empêche de présenter le pourcentage de latence comme une propriété stable du matériel. Les deux campagnes indiquent le même sens du compromis, sous leur charge séquentielle.

Le diagnostic suggère de comparer, lors d'un prochain protocole déclaré, une sortie contrainte par schéma ou une réparation déterministe à l'entraînement. Il ne prouve pas que le LoRA est nécessaire. Le score antérieur obtenu en comparant les probabilités de deux suffixes imposés ne se transpose pas directement à une génération libre.

## Vérification et preuves

**189 tests passent** (26,357 s). Ils couvrent notamment budgets et dépassements, seuils par tranche, directions des objectifs, échec d'un candidat, référence conservée, refus total, absence d'essai d'un second candidat sur le test final, empreintes modifiées et isolation entre profils. Le vrai Doctor Hermes enregistre 19 outils et neuf dispatchs sûrs passent hors du dépôt. La lecture native du rapport fonctionne dans le profil producteur et échoue depuis le profil destinataire.

- Première campagne : `.local/hermes-experiment/profiles/ai-lab-teacher/ai-lab/workspace/controller-v1-01/`.
- Rejeu instrumenté : même racine, `controller-v1-02/`. Chaque répertoire conserve `request.json`, `selection.json`, `trials.jsonl` et `report.json`.
- Contrôles recalculés : `.local/controller-v1/verification.json`, avec empreintes, ordre des essais, identité des sorties LoRA, paramètres effectifs et absence de mesures finales.
- Jobs et commandes Hermes : `.local/controller-v1/submission*.json`, `job-final*.json`, `progress*.json`, `native-report.json`, `native-cross-profile.json`, `native-registration.txt`.
- Suite finale : `.local/controller-v1/test-results-final.txt`.

Les deux exécutions ont réussi techniquement. Leur verdict produit reste **`no_qualified_configuration`**. Les 64 cas réservés restent disponibles pour la confirmation d'un candidat futur sélectionné sur développement ; toute nouvelle recette doit être figée avant leur évaluation. Ces essais ne valident ni déduplication automatique de mémoire, ni usage d'entreprise, ni kernels CUDA.

Statut ultérieur : ces 64 cas ont été consommés par la [comparaison base/LoRA avec JSON contraint](2026-11-09-json-comparison.md), après gel de cette nouvelle paire. La disponibilité indiquée ci-dessus décrit l'état à la fin de la version 0.5.
