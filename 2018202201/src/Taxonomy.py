import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS

df_citations = pd.read_csv("arxiv-metadata-ext-citation.csv", dtype={"id": object, "id_reference": object})
# print(df_citations.head())

df_categories = pd.read_csv("arxiv-metadata-ext-category.csv", dtype={"id": object})
# print(df_categories.head())

df_papers = pd.read_csv("arxiv-metadata-ext-paper.csv")
# print(df_papers.head())

df_versions = pd.read_csv("arxiv-metadata-ext-version.csv", dtype={'id': object})
# print(df_versions.head())

df_taxonomy = pd.read_csv("arxiv-metadata-ext-taxonomy.csv")
# print(df_taxonomy.groupby(['group_name', 'archive_name']).head(3))


def count_by_category_and_year(group_name):
    cats = df_categories.merge(df_taxonomy, on="category_id").query("group_name == @group_name").merge(
        df_versions.query("version =='v1'")[["id", "year"]], on="id")
    cats = cats.groupby(["year", "category_name"]).count().reset_index().pivot(index="category_name", columns="year",
                                                                               values="id")
    return cats


def count_by_archive_and_year(group_name):
    cats = df_categories.merge(df_taxonomy, on="category_id").query("group_name == @group_name").merge(
        df_versions.query("version =='v1'")[["id", "year"]], on="id")
    cats = cats.groupby(["year", "archive_name"]).count().reset_index().pivot(index="archive_name", columns="year",
                                                                              values="id")
    return cats


def show_count_by_category_and_year(group_name, figsize=(10, 5)):
    plt.figure(figsize=figsize)
    plt.title(f"{group_name} papers by category and year")
    sns.heatmap(count_by_category_and_year(group_name), cmap="Greens", linewidths=0.01, linecolor='palegreen')
    plt.show()


def show_count_by_archive_and_year(group_name="Physics", figsize=(10, 5)):
    plt.figure(figsize=figsize)
    plt.title(f"{group_name} papers by archive and year")
    sns.heatmap(count_by_archive_and_year(group_name), cmap="Greens", linewidths=0.01, linecolor='palegreen')
    plt.show()


def top_k_influential(group_name, top_k=5, threshold=100):
    ids = df_categories.merge(df_taxonomy, on="category_id").query("group_name ==@group_name")["id"].values
    cits = df_citations.query('id.isin(@ids)', engine="python").merge(
        df_versions.query("version == 'v1'")[["id", "year"]], on="id").groupby(["year", "id_reference"]).count()
    cits = cits.reset_index()
    cits = cits.loc[cits.groupby('year')['id'].nlargest(top_k).reset_index()['level_1']]
    cits = cits.query("id > @threshold")
    cits = cits.rename(columns={"id": "references", "id_reference": "id"})
    cits = cits.merge(df_papers, on="id")

    return cits


def show_influential_heatmap(group_name, cits, figsize=(10, 25)):
    hm_cits = cits.pivot(index="title", columns="year", values="references")

    plt.figure(figsize=figsize)
    plt.title("Top influential papers by year")
    sns.heatmap(hm_cits, cmap="Greens", linewidths=0.01, linecolor='palegreen')
    plt.show()


def make_clickable(val):
    # target _blank to open new window
    return '<a target="_blank" href="{}">{}</a>'.format(val, val)


def show_influential_table(cits):
    df = cits.groupby(["id", "title", "authors", "abstract"]).agg({"references": "sum"}).reset_index()
    df = df.sort_values(by="references", ascending=False).reset_index(drop=True)
    df["url"] = df["id"].map(lambda x: f'https://arxiv.org/pdf/{x}')
    df["authors"] = df["authors"].map(lambda x: x if len(str(x)) < 50 else str(x)[:47] + "...")

    df = df[["title", "authors", "abstract", "url", "references"]]
    return df.style.format({'url': make_clickable})


def plot_wordcloud(text, mask=None, max_words=200, max_font_size=100, figure_size=(24.0, 16.0),
                   title=None, title_size=40, image_color=False):
    stopwords = set(STOPWORDS)
    more_stopwords = {'We', 'paper', 'new'}
    stopwords = stopwords.union(more_stopwords)

    wordcloud = WordCloud(background_color='black',
                          stopwords=stopwords,
                          max_words=max_words,
                          max_font_size=max_font_size,
                          random_state=42,
                          width=800,
                          height=400,
                          mask=mask,
                          min_word_length=4,
                          # normalize_plurals = True,
                          # collocations = True,
                          # collocation_threshold = 10
                          )
    wordcloud.generate(str(text))

    plt.figure(figsize=figure_size)
    if image_color:
        image_colors = ImageColorGenerator(mask);
        plt.imshow(wordcloud.recolor(color_func=image_colors), interpolation="bilinear");
        plt.title(title, fontdict={'size': title_size,
                                   'verticalalignment': 'bottom'})
    else:
        plt.imshow(wordcloud);
        plt.title(title, fontdict={'size': title_size, 'color': 'black',
                                   'verticalalignment': 'bottom'})
    plt.axis('off');
    plt.tight_layout()


df = df_versions.query("version =='v1'").groupby(["year","month"]).agg({"id":'count'}).reset_index()
df["tot"] = df["id"].cumsum()

df = df.query("year > 1990 and ( year != 2020 or month < 8)")
df["month"] =  df["year"].astype(str) + "-" + df["month"].astype(str)


_df = df_categories.merge(df_taxonomy, on="category_id", how="left").drop_duplicates(["id","group_name"]).groupby("group_name").agg({"id":"count"}).sort_values(by="id",ascending=False).reset_index()

fig = plt.figure(figsize=(20,10))

ax1 = plt.subplot2grid((2, 2), (0, 0))
ax1.title.set_text('ArXiv papers')
ax1.plot(df["month"], df["tot"])
ax1.hlines(y=1e6, xmin=0, xmax=len(df), color='green', linestyle="dotted")
ax1.hlines(y=1.5e6, xmin=0, xmax=len(df), color='green', linestyle="dotted")
ax1.set_xticks(np.arange(0, len(df), 12.0))
ax1.tick_params('x',labelrotation=90)


ax2 = plt.subplot2grid((2, 2), (1, 0))
ax2.title.set_text("ArXiv papers by month")
ax2.plot(df["month"], df["id"])
ax2.hlines(y=10000, xmin=0, xmax=len(df), color='green', linestyle="dotted")
ax2.hlines(y=15000, xmin=0, xmax=len(df), color='green', linestyle="dotted")
ax2.set_xticks(np.arange(0, len(df), 12.0))
ax2.tick_params('x',labelrotation=90)

ax3 = plt.subplot2grid((2, 2), (0, 1), rowspan=2)
ax3.title.set_text("ArXiv papers by group")
explode = (0, 0, 0, 0.2, 0.3, 0.3, 0.2, 0.1)
ax3.pie(_df["id"],  labels=_df["group_name"], autopct='%1.1f%%', startangle=160, explode=explode)

plt.tight_layout()
plt.show()

df = df_citations.query("id_reference == '1412.6980'")\
.merge(df_categories,on="id")\
.merge(df_taxonomy,on="category_id").drop_duplicates(["id","group_name"])\
.merge(df_versions.query("version =='v1'")[["id","year"]], on ="id")


hmap =df.groupby(["group_name","year"]).agg({"id":"count"}).reset_index().pivot(index=["group_name"], columns="year", values="id")

plt.figure(figsize=(10,5))
plt.title("Papers that reference 'Adam: A Method for Stochastic Optimization'")
sns.heatmap(hmap,cmap="Greens", linewidths=0.01, linecolor='palegreen', annot=True, fmt=".0f")
plt.show()