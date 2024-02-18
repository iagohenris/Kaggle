library(tidyverse)
library(dplyr)
library(gmodels)
library(pheatmap)
library(rpart)


#código previamente feito em python, refazendo em R pontos cruciais
train <- read.csv('inputs\\train.csv')
test <- read.csv('inputs\\test.csv')

df_merged <- bind_rows(train, test)
#substituindo nulos na idade

df_merged$Age[is.na(df_merged$Age)] <- median(df_merged$Age, na.rm = TRUE)
#substituindo nulos na tarifa
df_merged$Fare[is.na(df_merged$Fare)] <- median(df_merged$Fare, na.rm = TRUE)
#alterando sexo para 0 - masculino e 1 - feminino
mapeamento <- c("male" = 0, "female" = 1)
df_merged$Sex <- mapeamento[df_merged$Sex]
#separando novamente treino e teste
train_trat <- df_merged[1:nrow(train),]
test_trat <- df_merged[(nrow(train)+1):nrow(df_merged),]
#regressão logistica
modelo_log <- glm(formula = Survived ~ Pclass + Sex + Fare + Age + SibSp + Parch, data = train_trat, family = binomial)
resultados_log <- predict(modelo_log, test_trat, type = "response")
#transformando em binario
resultados_log <- ifelse(resultados_log > 0.5, 1, 0)
final_log <- test %>% select(PassengerId) %>% mutate(Survived = resultados_log)
#arvore de decisao
modelo_arvore <- rpart(Survived ~ Pclass + Sex + Fare + Age + SibSp + Parch, data = train_trat, method = "class")
# Fazer previsões no conjunto de teste
resultados_arvore <- predict(modelo_arvore, newdata = test_trat, type = "class")
final_arvore <- test %>% select(PassengerId) %>% mutate(Survived = resultados_arvore)
# Avaliar a precisão do modelo
#matriz confusão com modelo de treinamento
treino_log <- predict(modelo_log, train_trat, type = "response")
treino_log <- ifelse(treino_log > 0.5, 1, 0)
matriz_confusao_lg <- table(train_trat$Survived, treino_log)
#matriz confusão com modelo de treinamento
treino_arvore <- predict(modelo_arvore, train_trat, type = "class")
matriz_confusao_av <- table(train_trat$Survived, treino_arvore)
#salvando arquivos
write.csv2(final_log, "outputs/r_result_lr.csv", row.names = FALSE)
write.csv2(final_arvore, "outputs/r_result_tree.csv", row.names = FALSE)
