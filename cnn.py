this is the line added by satyasai!!!!!!!!!!!!!!!!!!!
from gensim.models import Doc2Vec
import multiprocessing

cores = multiprocessing.cpu_count()
model_dbow = Doc2Vec(dm=0, size=100, negative=5, min_count=2, workers=cores, alpha=0.065, min_alpha=0.065)
model_dbow.build_vocab([x for x in tqdm(all_x_w2v)])
model_dbow.train(utils.shuffle([x for x in tqdm(all_x_w2v)]), total_examples=len(all_x_w2v), epochs=1)

def build_doc_Vector(tokens, size):
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in tokens:
        try:
            vec += model_dbow[word].reshape((1, size)) * tfidf[word]
            count += 1.
        except KeyError: 
            continue
    if count != 0:
        vec /= count
    return vec

train_vecs_dbow = np.concatenate([build_doc_Vector(z, 100) for z in tqdm(map(lambda x: x.words, x_train))])
train_vecs_dbow = scale(train_vecs_dbow)
val_vecs_dbow = np.concatenate([build_doc_Vector(z, 100) for z in tqdm(map(lambda x: x.words, x_validation))])
val_vecs_dbow = scale(val_vecs_dbow)

model = Sequential()
model.add(Dense(256, activation='relu', input_dim=100))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(train_vecs_dbow, y_train, epochs=100, batch_size=32, verbose=2)
score,acc = model.evaluate(val_vecs_dbow, y_validation, batch_size=128, verbose=2)
print("score: %.2f" % (score))
print("acc: %.2f" % (acc))
#print (score[1])