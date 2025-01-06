res = read.csv("MS_wavelet_reserach/final_result.csv") %>% 
  mutate(wsize= 2^J.Setting) |> 
  mutate(accuracy= accuracy_test, 
         proc=Preprocess,
         proc = recode(proc, preprocess1 = 'PROC1', 
                       preprocess2 = 'PROC2',
                       preprocess3 = 'PROC3',
                       preprocess4 = 'PROC4'),
         Oversample = recode(Oversample, 
                             "0" = 'Nosampling', 
                             "1" = 'Upsampling'),
         across(where(is.character),as.factor)) |> 
  select(res_id, model, wavelet, Oversample, proc, wsize, precision, recall, F1.score, accuracy) 



res |> 
  select(model, proc, precision, recall, F1.score, accuracy) |> 
  pivot_longer(cols=-c(model,proc), names_to = "metric", values_to = "value") %>% 
  ggplot(aes(x=proc,y=value, col=model,fill=metric)) +
  #geom_point()+
  stat_summary(geom = "point",
               fun = mean,
               size=4)+
  labs(y= "Average value")+
  theme_apa()
  #facet_wrap(~metric)

res %>% 
  select(model,precision, recall, F1.score, accuracy_test, Preprocess) %>%
  pivot_longer(cols=-c(model,Preprocess), names_to = "metric", values_to = "value") -> long_res

long_res %>% 
  ggplot(aes(x=metric ,y=value, group=Preprocess), position = jitter(x = metric)) +
  geom_point(size=3)+
  theme_apa()

res %>% 
  mutate(accuracy= accuracy_test, 
         proc=Preprocess) %>% 
  mutate(proc = recode(proc, preprocess1 = 'PROC1', 
                       preprocess2 = 'PROC2',
                       preprocess3 = 'PROC3',
                       preprocess4 = 'PROC4')) %>% 
  select(precision, recall, F1.score, accuracy, proc) %>%

  tbl_summary(
    by = proc,
    type = list(
      c("recall","accuracy") ~ 'continuous'),
    statistic = list(all_continuous() ~ "{mean} ± {sd} ({min},{max})"),
    digits = all_continuous() ~ 2
  ) %>% 
  add_p() %>%
  bold_p() %>% 
  #add_overall %>%
  bold_labels 

