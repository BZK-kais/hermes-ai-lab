# Contrat de livraison — AI Lab 1.0 locale

Fixé le 11 novembre 2026, avant la vérification de cette livraison. L'utilisateur demande de terminer le plugin de bout en bout. La cible vérifiable est un laboratoire natif Hermes sur le Mac disponible, avec interfaces CLI, MCP et HTTP partageant les mêmes contrats. Les intégrations NVIDIA et fournisseurs existantes doivent rester utilisables selon leurs dépendances et annoncer leur niveau de vérification réel.

## Parcours obligatoires

1. Distribuer une archive propre, installable dans un profil Hermes neuf sans dépendre du checkout de développement, et un wheel standalone installable sans mode éditable. Aucun état privé, secret ou poids local dans la distribution.
2. Initialiser le profil et déclarer les chemins locaux par la commande d'installation/configuration. Vérifier les capacités sans téléchargement, entraînement ni activation implicite. Fournir un skill natif qui guide le parcours conversationnel.
3. Transformer une mission complète, ses seuils et budgets en un protocole borné. Renvoyer les informations manquantes pour une mission incomplète. Ne pas inventer les seuils ni décoder les références du test pendant la préparation.
4. Effectuer une comparaison réelle via Hermes sur le Mac. Distinguer réussite technique, référence conservée, amélioration qualifiée et rejet. Le test fonctionnel de livraison ne constitue pas une nouvelle démonstration de gain LoRA sur PAWS.
5. Capturer une configuration acceptée avec sa preuve locale et ses empreintes. Refuser une preuve rejetée, étrangère ou altérée. Activer explicitement, générer avec la configuration mesurée, et revenir à l'état précédent.
6. Exporter puis importer un paquet portable. Un import reste inactif et demande une nouvelle qualification locale. Réutiliser les poids de base par une référence et une empreinte, sans les inclure implicitement dans le paquet.
7. Exposer progression et checkpoints de l'entraînement MLX général. Vérifier un entraînement court et le rechargement de son adaptateur ; ceci est une preuve de fonctionnement, pas une preuve d'amélioration qualitative.
8. Exécuter les tests pertinents, les dispatchs du SDK Hermes installé, les transports CLI/MCP/HTTP et les essais de portabilité. Documenter les commandes, artefacts livrés, résultats et limites.

## Limites de la preuve

La RTX 4070 n'est toujours pas accessible ; aucun DGX Spark, RTX Blackwell ni compte de calcul payant n'a été fourni pour exécution. Leurs connecteurs peuvent être livrés et testés sur contrats, sans annoncer une certification matérielle. Aucun service externe ne sera publié ni ressource payante créée dans ce lot. Le serving concurrent, les tâches de retrieval et les évaluateurs sémantiques universels ne sont pas démontrés par le contrôleur MLX séquentiel.

Les anciens résultats PAWS restent inchangés et non qualifiés. Leurs corpus finaux consommés et leurs snapshots de sources sont conservés. Les nouveaux parcours de livraison utilisent des fixtures distinctes et leurs résultats ne remplacent pas cette conclusion.
