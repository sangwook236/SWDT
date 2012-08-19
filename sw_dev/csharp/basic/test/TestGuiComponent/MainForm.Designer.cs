namespace TestGuiComponent
{
    partial class MainForm
    {
        /// <summary>
        /// 필수 디자이너 변수입니다.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// 사용 중인 모든 리소스를 정리합니다.
        /// </summary>
        /// <param name="disposing">관리되는 리소스를 삭제해야 하면 true이고, 그렇지 않으면 false입니다.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form 디자이너에서 생성한 코드

        /// <summary>
        /// 디자이너 지원에 필요한 메서드입니다.
        /// 이 메서드의 내용을 코드 편집기로 수정하지 마십시오.
        /// </summary>
        private void InitializeComponent()
        {
            this.openFileDialogButton = new System.Windows.Forms.Button();
            this.saveFileDialogButton = new System.Windows.Forms.Button();
            this.printDialogButton = new System.Windows.Forms.Button();
            this.SuspendLayout();
            // 
            // openFileDialogButton
            // 
            this.openFileDialogButton.Location = new System.Drawing.Point(32, 15);
            this.openFileDialogButton.Name = "openFileDialogButton";
            this.openFileDialogButton.Size = new System.Drawing.Size(137, 24);
            this.openFileDialogButton.TabIndex = 0;
            this.openFileDialogButton.Text = "Open File Dialog...";
            this.openFileDialogButton.UseVisualStyleBackColor = true;
            this.openFileDialogButton.Click += new System.EventHandler(this.openFileDialogButton_Click);
            // 
            // saveFileDialogButton
            // 
            this.saveFileDialogButton.Location = new System.Drawing.Point(196, 15);
            this.saveFileDialogButton.Name = "saveFileDialogButton";
            this.saveFileDialogButton.Size = new System.Drawing.Size(137, 24);
            this.saveFileDialogButton.TabIndex = 1;
            this.saveFileDialogButton.Text = "Save File Dialog...";
            this.saveFileDialogButton.UseVisualStyleBackColor = true;
            this.saveFileDialogButton.Click += new System.EventHandler(this.saveFileDialogButton_Click);
            // 
            // printDialogButton
            // 
            this.printDialogButton.Location = new System.Drawing.Point(360, 15);
            this.printDialogButton.Name = "printDialogButton";
            this.printDialogButton.Size = new System.Drawing.Size(137, 24);
            this.printDialogButton.TabIndex = 2;
            this.printDialogButton.Text = "Print Dialog...";
            this.printDialogButton.UseVisualStyleBackColor = true;
            this.printDialogButton.Click += new System.EventHandler(this.printDialogButton_Click);
            // 
            // MainForm
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(7F, 12F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(528, 311);
            this.Controls.Add(this.printDialogButton);
            this.Controls.Add(this.saveFileDialogButton);
            this.Controls.Add(this.openFileDialogButton);
            this.Name = "MainForm";
            this.Text = "Windows Form";
            this.ResumeLayout(false);

        }

        #endregion

        private System.Windows.Forms.Button openFileDialogButton;
        private System.Windows.Forms.Button saveFileDialogButton;
        private System.Windows.Forms.Button printDialogButton;

    }
}

