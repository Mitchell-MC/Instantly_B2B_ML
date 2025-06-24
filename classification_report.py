# In main(), after model.fit()

from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Make predictions on the test set
y_pred = model.predict(X_test)

# Print the classification report
print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred, target_names=['No Engagement (0)', 'Opener (1)', 'Clicker (2)']))

# Generate and save the confusion matrix
print("\n--- Confusion Matrix ---")
cm = confusion_matrix(y_test, y_pred)
print(cm)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['No Engagement', 'Opener', 'Clicker'],
            yticklabels=['No Engagement', 'Opener', 'Clicker'])
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.savefig('confusion_matrix.png')
print("Confusion matrix plot saved as 'confusion_matrix.png'")