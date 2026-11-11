# Déduplication de faits courts avec Hermes AI Lab

Cette expérience mesure la capacité d’un petit modèle local à reconnaître que deux phrases anglaises expriment le même fait. Dans un futur flux mémoire, une recherche lexicale pourrait retrouver les notes proches, puis le classifieur signaler les paires à examiner pour une fusion. Par exemple, « Mia replaced Jo on Monday » et « Jo replaced Mia on Monday » contiennent les mêmes mots mais inversent les rôles. Une fusion supprimerait une distinction utile. Le test porte sur la classification de paires. L’intégration à la mémoire Hermes et le comportement d’un système de fusion restent à évaluer.

L’[expérience de triage](2026-11-09-triage-v1.md) avait montré qu’un Naïve Bayes lexical suffisait à battre le petit modèle adapté sur ce corpus synthétique. PAWS-Wiki cible des paires où le recouvrement lexical reste élevé même si le sens change. L’article historique rapporte 55,8 % pour son modèle sac de mots et 90,4 % pour son BERT supervisé sur le test PAWS-Wiki. Ces résultats situent la difficulté du benchmark et utilisent d’autres modèles et volumes d’entraînement que cet essai. [Article PAWS](https://aclanthology.org/N19-1131/)

## Données et comparaison

PAWS-Wiki Labeled Final part de phrases Wikipédia et construit des paires par permutation de mots et rétrotraduction. Des annotateurs humains jugent la fluidité et la paraphrase. Les trois Parquet épinglés contiennent 49 401 paires d’entraînement, 8 000 de validation et 8 000 de test. Google autorise l’usage libre et invite à créditer la source. [Dépôt Google Research](https://github.com/google-research-datasets/paws) · [Licence](https://github.com/google-research-datasets/paws/blob/master/LICENSE) · [Révision et empreintes Parquet](https://huggingface.co/datasets/google-research-datasets/paws/commit/161ece9501cf0a11f3e48bd356eaa82de46d6a09)

L’échantillonnage déterministe utilise la graine 20261109 et choisit une paire par groupe de phrases normalisées reliées. Il produit quatre jeux équilibrés entre `same` et `different`, avec 1 024 paires pour l’entraînement, 64 pour le développement et 128 pour chacun des deux profils. Les deux jeux finaux proviennent du test officiel. L’audit a trouvé zéro groupe traversant les splits sources et zéro collision entre les jeux échantillonnés. Les empreintes couvrent les quatre jeux, le contrat, les prompts, le protocole et la provenance. [Provenance](../../examples/experiments/pair-paraphrase-v1/provenance.json) · [Manifeste](../../examples/experiments/pair-paraphrase-v1/manifest.sha256)

- **Base locale** — Qwen2.5-1.5B-Instruct MLX, 4 bits. Deux prompts avec deux exemples issus de l’entraînement chacun sont comparés sur développement. [Poids épinglés](https://huggingface.co/mlx-community/Qwen2.5-1.5B-Instruct-4bit/commit/8b403126fc14f14cfc99bb4cfa72ecbc129ea677)
- **Adaptation** — deux recettes LoRA sur la base quantifiée. La première utilise 128 itérations, 8 couches et un taux de 2e-4. La seconde utilise 256 itérations, 12 couches et un taux de 1e-4. Les deux ont un rang 16, un batch de 4, une limite de 1 024 tokens et une perte limitée à la réponse. La sélection sur développement privilégie le nombre de réponses correctes, puis le plus faible nombre de fausses fusions et l’ordre de déclaration. La base et l’adaptateur retenu utilisent le même prompt et le même décodage par log-vraisemblance des deux sorties JSON canoniques. [Documentation MLX](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md)
- **Baseline CPU** — TF-IDF mots et caractères, recouvrement de mots et de bigrammes, similarité d’ordre et d’édition, concordance des nombres et négations. Les deux configurations utilisent les mêmes 1 024 paires. Le gradient boosting a été figé après 47/64 sur développement avec 10 fausses fusions, contre 43/64 et 12 pour la régression logistique. [Script CPU](../../scripts/evaluate_pair_baselines.py)

Le protocole exige sur **chacun** des deux jeux finaux au moins 104/128 réponses correctes, huit corrections nettes de plus que la base sans adaptateur, quatre réponses correctes de plus que la baseline CPU et aucune hausse des fausses fusions par rapport à la base sans adaptateur. Une fausse fusion est une paire `different` prédite `same`. Son taux se calcule sur les 64 paires `different` du jeu. [Protocole figé](../../examples/experiments/pair-paraphrase-v1/protocol.json)

## Résultats réservés

Le job producteur initial `2cd3d228-4d03-46f9-8a8c-58caf8f3ea07`, soumis par le registre Hermes, a terminé le premier entraînement puis échoué pendant l'évaluation sur développement, avant toute sélection et tout test final. Le diagnostic sur des exemples d'entraînement a observé un cache Metal de plus de 24 Go, que le compteur de pic mémoire active MLX ne reflétait pas.

Le worker fixe désormais une cible de 256 Mio pour le cache libre MLX, dans ses processus d'entraînement et d'inférence. Cette cible n'est pas un plafond de mémoire totale du Mac. Une vérification sur 64 exemples d'entraînement passe après correction ; les scores des deux premiers exemples sont identiques à la vérification précédente. Le protocole a été relancé intégralement par la commande native `hermes ai-lab experiment`, job `b0fbf7bb-3dfe-49f2-ac5c-3bc7b8b95cc8`. Aucun exemple, seuil, prompt ou hyperparamètre d'entraînement n'a changé. [Gel initial](../../.local/pair-paraphrase/protocol-freeze.json) · [Révision technique du gel](../../.local/pair-paraphrase/protocol-freeze-runtime-fix.json) · [Vérification du cache](../../.local/pair-paraphrase/bounded64-summary.json)

La sélection a retenu le prompt `equivalence` (44/64 sur développement, contre 40/64 pour l'autre prompt) et la deuxième recette LoRA (51/64, contre 32/64 pour la première). La première recette répondait `same` sur tous les cas de développement. Le candidat retenu et les empreintes ont été écrits dans `selected_prompt.json` et `selection.json` avant d'ouvrir le test producteur.

| Solution | Producteur, 128 paires | Destinataire, 128 autres paires |
| --- | ---: | ---: |
| Référence CPU entraînée | 79 — 61,7 % | 93 — 72,7 % |
| Qwen sans adaptateur, meilleur prompt | 88 — 68,8 % | 98 — 76,6 % |
| Même Qwen et prompt, avec LoRA retenu | **107 — 83,6 %** | **103 — 80,5 %** |
| Fausses équivalences : base → LoRA | 34 → 2 | 23 → 4 |
| Vrais doublons manqués : base → LoRA | 6 → 19 | 7 → 21 |
| Critères préenregistrés | **Réussis** | **Échec** |

Le destinataire manque deux exigences : 103 bonnes réponses au lieu des 104 requises et cinq corrections nettes au lieu de huit. Les deux jobs sont techniquement réussis, mais la qualification complète exige les deux jeux. **Aucune activation n'a été effectuée.** Les 256 cas finaux sont maintenant consommés et ne doivent pas servir à choisir une nouvelle recette présentée comme indépendante.

Le modèle corrige 33 erreurs et en introduit 14 sur le producteur ; il en corrige 19 et en introduit 14 sur le destinataire. Le test exact apparié de McNemar donne respectivement p = 0,00794 et p = 0,48685 : le second gain de précision reste incertain à cette taille d'échantillon. Les intervalles de Wilson et le détail des comparaisons sont conservés dans le [résumé calculé](../../.local/pair-paraphrase/result.json). Ce sont des descriptions des résultats, pas de nouvelles règles de qualification.

L'adaptateur et son prompt ont été copiés dans le second profil après le test producteur ; les empreintes de modèle, adaptateur et prompt concordent. Le jeu reçu contient ses 128 cas, les métadonnées et le manifeste, sans fichier d'entraînement ni de développement. Le prompt conserve ses deux exemples. La baseline CPU locale figée est également disponible dans les deux profils. [Preuve de transfert](../../.local/pair-paraphrase/transfer-proof.json)

Le parcours réussi a duré 716,454 s côté producteur, dont 172,362 s et 367,947 s pour les deux entraînements, puis 82,763 s côté destinataire. Ces temps excluent téléchargement, préparation et diagnostic de l'échec initial. L'adaptateur retenu et sa configuration occupent 31 673 464 octets. Le pic de mémoire **active MLX** pendant son entraînement est de 3 858 300 712 octets ; ce n'est pas une mesure de RAM totale incluant cache et système. Les latences médianes par paire sont de 287/347 ms (base/LoRA) côté producteur et 281/334 ms côté destinataire, hors démarrage et chargement. Ce passage unique ne constitue pas un benchmark de débit.

## Contrôle complémentaire du seuil, après le test

Une baisse des fausses équivalences peut provenir d'un modèle devenu simplement plus conservateur. Pour examiner ce point, une analyse supplémentaire a calibré un seuil du modèle initial sur les **64 cas de développement seulement**, en maximisant les bonnes réponses puis en minimisant les fausses équivalences. Cette analyse a été décidée après lecture des résultats finaux : elle est **exploratoire**, distincte du protocole préenregistré et ne modifie aucun verdict.

Le seuil de marge `score(same) − score(different)` choisi est 3,067975, avec 50/64 sur développement. Appliqué sans autre ajustement, le modèle initial calibré obtient **98/128 sur chacun des deux tests**, contre 107/128 et 103/128 pour le LoRA. Ses fausses équivalences sont 13 et 8, contre 2 et 4 ; ses vrais doublons manqués sont 17 et 22, contre 19 et 21. Ainsi, sur les deux jeux réunis, on observe **21 → 6 fausses équivalences et 39 → 40 doublons manqués**. L'aire sous la courbe ROC progresse aussi, de 0,845 à 0,935 et de 0,861 à 0,904. Ces observations suggèrent un apport au-delà d'un décalage de seuil, sans remplacer une confirmation indépendante. [Analyse exploratoire et provenance](../../.local/pair-paraphrase/calibration-audit.json)

## Preuves et reprise

- [Rapport producteur](../../.local/hermes-experiment/profiles/ai-lab-teacher/ai-lab/workspace/pair-paraphrase-v1-02/report.json), job `b0fbf7bb-3dfe-49f2-ac5c-3bc7b8b95cc8`.
- [Rapport destinataire](../../.local/hermes-experiment/profiles/ai-lab-receiver/ai-lab/workspace/pair-paraphrase-v1-receiver-01/report.json), job `dc645941-030b-4fd3-8214-8d882eb97aeb`.
- [Provenance des exécutions et versions](../../.local/pair-paraphrase/run-record.json), [isolation logique des profils](../../.local/pair-paraphrase/profile-isolation.json), [137 tests réussis](../../.local/pair-paraphrase/test-results-final.txt).
- [Lecture du rapport par le CLI natif](../../.local/pair-paraphrase/native-report-receiver.json), avec exemples détaillés exclus de la réponse de l'outil.

Prochaine confirmation proposée : figer avant tout essai une baseline calibrée et des notes représentatives en français, avec des coûts explicites pour fausse équivalence et doublon manqué. Le présent candidat reste un filtre expérimental de propositions de déduplication ; aucune mémoire Hermes n'est fusionnée ou supprimée par cette expérience.

## Portée

Les profils producteur et receveur sont séparés dans Hermes et utilisent le même Mac M5 de 32 Go. Le deuxième jeu apporte une mesure sur des paires distinctes. La reproduction sur une autre machine reste à faire. L’audit vérifie les correspondances exactes après normalisation. La présence éventuelle d’exemples PAWS dans le préentraînement du modèle reste inconnue.

L’essai mesure la classification de phrases anglaises courtes dans un échantillon équilibré de PAWS-Wiki. La fréquence des doublons dans une mémoire réelle peut différer de cet équilibre et changer la précision des propositions de fusion. Le transfert vers des notes Hermes, des textes plus longs ou le français demande une évaluation séparée.
